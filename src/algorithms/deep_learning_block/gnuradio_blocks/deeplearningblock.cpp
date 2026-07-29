#include "deeplearningblock.h"
#include "gnss_sdr_filesystem.h"
#include "onnx_model.h"
#include "stft_input.h"
#include <gnuradio/io_signature.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstdint>
#include <ctime>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

// command_id values for event_type=30 (set_input_filter), same as tcp_cmd_interface.cc
static constexpr int FILTER_CMD_PASS_THROUGH    = 301;
static constexpr int FILTER_CMD_NOTCH           = 302;
static constexpr int FILTER_CMD_PULSE_BLANKING  = 303;
static constexpr int FILTER_CMD_NOTCH_LITE      = 304;

namespace {
// Parametry STFT -- MUSZA byc identyczne jak przy generowaniu danych treningowych
// (03_Kod/modules/jammers/Constants.py: Nps_n=128 (nperseg), OVERLAP=0.75,
// WINDOW_FUNCTION="hann"; patrz tez GenerateDatasets.py).
// Sam STFT + normalizacja licza sie w compute_stft_input()/normalize_mag_db() (modules/neural_network/include/stft_input.h), zeby nie utrzymywac dwoch kopii tej samej logiki i nie ryzykowac ich rozjechania.
constexpr int kStftNperseg = 128;
constexpr double kStftOverlap = 0.75;
constexpr double kStftClipDb = 60.0;

std::string iso_timestamp_now()
{
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto ms = duration_cast<milliseconds>(now.time_since_epoch()).count() % 1000;
    const std::time_t t = system_clock::to_time_t(now);
    std::tm tm_utc{};
    gmtime_r(&t, &tm_utc);
    std::ostringstream oss;
    oss << std::put_time(&tm_utc, "%Y-%m-%dT%H:%M:%S") << '.' << std::setfill('0') << std::setw(3) << ms << 'Z';
    return oss.str();
}

const char *jammer_type_name(JammerType jammer)
{
    switch (jammer)
        {
        case JammerType::CHIRP:
            return "Chirp";
        case JammerType::CW:
            return "CW";
        case JammerType::FM:
            return "FM";
        case JammerType::NARROWBAND:
            return "Narrowband";
        case JammerType::NOJAM:
            return "No Jam";
        case JammerType::PULSED:
            return "Pulsed";
        }
    return "Unknown";
}

const char *filter_name(int command_id)
{
    switch (command_id)
        {
        case FILTER_CMD_PASS_THROUGH:
            return "Pass_Through";
        case FILTER_CMD_NOTCH:
            return "Notch";
        case FILTER_CMD_PULSE_BLANKING:
            return "Pulse_Blanking";
        case FILTER_CMD_NOTCH_LITE:
            return "Notch_Lite";
        }
    return "Unknown";
}
}  // namespace


class DeepLearningBlock::Opaque {
public:
    Opaque(DeepLearningBlock *parent,
           int window_size,
           const std::string &model_path,
           const std::string &log_dir,
           std::shared_ptr<Concurrent_Queue<pmt::pmt_t>> queue)
        : window_size_(window_size),
          control_queue_(std::move(queue)),
          p_(parent)
    {
        if (window_size_ < kStftNperseg)
            throw std::invalid_argument(
                "DeepLearningBlock: window_size must be >= " + std::to_string(kStftNperseg));

        const fs::path dir(log_dir);
        fs::create_directories(dir);
        spectrogram_preview_path_ = (dir / "spectrogram_live.pgm").string();

        inference_log_.open((dir / "inference_log.csv").string(), std::ios::trunc);
        inference_log_ << "timestamp_utc,stft_latency_ms,onnx_latency_ms,total_latency_ms,jammer_type\n";

        filter_log_.open((dir / "filter_switch_log.csv").string(), std::ios::trunc);
        filter_log_ << "timestamp_utc,from_filter,to_filter\n";

        const OrtPathString ort_model_path(model_path.begin(), model_path.end());
        om_ = new OnnxModel(ort_model_path);

        // Inferencja idzie w osobnym wątku, zeby ciezka praca (STFT + ONNX)
        // nigdy nie blokowala wątku schedulera GNU Radio (a przez to bufora
        // wspoldzielonego z reszta flowgraphu - filtrem, trackingiem itd.)
        worker_thread_ = std::thread(&Opaque::worker_loop, this);
    }

    ~Opaque()
    {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            stop_ = true;
        }
        cv_.notify_one();
        if (worker_thread_.joinable())
            worker_thread_.join();
        delete om_;
    }

    int window_size_;
    std::deque<gr_complex> buffer_;
    std::shared_ptr<Concurrent_Queue<pmt::pmt_t>> control_queue_;

    // true gdy worker_thread_ aktualnie przetwarza okno - wtedy general_work()
    // odrzuca (drop) nowe okno zamiast czekac, zeby nie zatrzymywac realtime sciezki.
    std::atomic<bool> busy_{false};
    std::mutex mtx_;
    std::condition_variable cv_;
    std::vector<gr_complex> pending_window_;
    bool has_work_{false};
    bool stop_{false};

    int last_command_id_{FILTER_CMD_PASS_THROUGH};
    std::string spectrogram_preview_path_;
    std::ofstream inference_log_;
    std::ofstream filter_log_;

    StftInput compute_spectrogram(const std::vector<gr_complex> &samples);
    JammerType run_inference(const StftInput &spectrogram);
    void send_filter_command(JammerType jammer_type);
    void dispatch(const std::deque<gr_complex> &window);

private:
    void worker_loop();
    void write_spectrogram_preview(const StftInput &spectrogram);
    void log_inference(const char *jammer_str, double stft_ms, double onnx_ms, double total_ms);

    DeepLearningBlock *p_{nullptr};
    OnnxModel *om_{nullptr};
    std::thread worker_thread_;
};


StftInput DeepLearningBlock::Opaque::compute_spectrogram(
    const std::vector<gr_complex> &samples)
{
    // gr_complex JEST std::complex<float> w GNU Radio, wiec to zwykla kopia,
    // nie konwersja. compute_stft_input() robi STFT (okno Hanna periodyczne,
    // fftshift) ORAZ normalizacje (odjecie maksimum, clip do [-kStftClipDb,0],
    // skala do [0,1]) -- ten sam wzor co Stft.go_stft()/GenerateDatasets.py w
    // treningu Pythonowym (JammerSTFTDataset), wiec wejscie do sieci jest tu
    // gwarantowane identyczne jak przy treningu, bez potrzeby generowania/
    // skalowania zadnego obrazu.
    const std::vector<std::complex<float>> iq(samples.begin(), samples.end());
    StftInput spectrogram = compute_stft_input(
        iq, static_cast<int>(samples.size()), kStftNperseg, kStftOverlap, kStftClipDb);

    write_spectrogram_preview(spectrogram);
    return spectrogram;
}


void DeepLearningBlock::Opaque::write_spectrogram_preview(const StftInput &spectrogram)
{
    // Nadpisywany plik PGM z najnowszym oknem STFT - do podgladu na zywo
    // (np. `feh --reload 1 spectrogram_live.pgm`). Pisanie do pliku tymczasowego
    // + rename jest atomowe, zeby viewer nigdy nie zlapal pol-zapisanego pliku.
    // spectrogram.data jest juz znormalizowany do [0,1] (patrz compute_stft_input),
    // wiec do PGM (0-255) trzeba go tylko przeskalowac z powrotem.
    const std::string tmp_path = spectrogram_preview_path_ + ".tmp";
    std::ofstream f(tmp_path, std::ios::binary | std::ios::trunc);
    if (!f)
        return;

    f << "P5\n" << spectrogram.n_frames << " " << spectrogram.n_freq << "\n255\n";
    std::vector<uint8_t> pixels(spectrogram.data.size());
    for (size_t i = 0; i < spectrogram.data.size(); ++i)
        pixels[i] = static_cast<uint8_t>(std::clamp(spectrogram.data[i] * 255.0f, 0.0f, 255.0f));
    f.write(reinterpret_cast<const char *>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
    f.close();

    errorlib::error_code ec;
    fs::rename(tmp_path, spectrogram_preview_path_, ec);
    if (ec)
        std::cerr << "DeepLearningBlock: could not update spectrogram preview: " << ec.message() << "\n";
}


JammerType DeepLearningBlock::Opaque::run_inference(
    const StftInput &spectrogram)
{
    // Kolejnosc klas modelu = CLASSES z models/ResNet18.ipynb 
    // Musi 1:1 odpowiadac JammerType (patrz deeplearningblock.h).
    static constexpr std::array<JammerType, 6> kClassToJammer = {
        JammerType::CW,
        JammerType::FM,
        JammerType::CHIRP,
        JammerType::PULSED,
        JammerType::NARROWBAND,
        JammerType::NOJAM,
    };

    const std::vector<int64_t> input_shape = {1, 1, spectrogram.n_freq, spectrogram.n_frames};
    const auto output = om_->run(spectrogram.data, input_shape);

    const auto max_it = std::max_element(output.begin(), output.end());
    const auto class_index = static_cast<size_t>(std::distance(output.begin(), max_it));

    return (class_index < kClassToJammer.size()) ? kClassToJammer[class_index] : JammerType::NOJAM;
}


void DeepLearningBlock::Opaque::send_filter_command(JammerType jammer_type)
{
    int command_id = FILTER_CMD_PASS_THROUGH;
    switch (jammer_type)
        {
        case JammerType::CW:
        case JammerType::FM:
        case JammerType::CHIRP:
            command_id = FILTER_CMD_NOTCH;
            break;
        case JammerType::PULSED:
            command_id = FILTER_CMD_PULSE_BLANKING;
            break;
        case JammerType::NARROWBAND:
        case JammerType::NOJAM:
            command_id = FILTER_CMD_PASS_THROUGH;
            break;
        }

    if (command_id != last_command_id_)
        {
            if (filter_log_.is_open())
                {
                    filter_log_ << iso_timestamp_now() << ','
                                << filter_name(last_command_id_) << ','
                                << filter_name(command_id) << '\n';
                    filter_log_.flush();
                }
            last_command_id_ = command_id;
        }

    if (!control_queue_)
        return;

    // identyczne z TcpCmdInterface::set_input_filter(), event_type=30
    const command_event_sptr new_evnt = command_event_make(command_id, 30);
    control_queue_->push(pmt::make_any(new_evnt));
}


void DeepLearningBlock::Opaque::log_inference(const char *jammer_str, double stft_ms, double onnx_ms, double total_ms)
{
    if (!inference_log_.is_open())
        return;
    inference_log_ << iso_timestamp_now() << ',' << stft_ms << ',' << onnx_ms << ',' << total_ms << ',' << jammer_str << '\n';
    inference_log_.flush();
}


void DeepLearningBlock::Opaque::dispatch(const std::deque<gr_complex> &window)
{
    // Caller (general_work()) only invokes this once it has already observed
    // busy_ == false, so this check is a defensive fallback against the
    // (expected to be essentially impossible, single-producer) race where
    // busy_ flips true again between that check and this call -- if it ever
    // fires it's worth knowing about, hence still logged.
    if (busy_.load(std::memory_order_acquire))
        {
            std::cerr << "DeepLearningBlock: worker unexpectedly busy, dropping window\n";
            return;
        }

    {
        std::lock_guard<std::mutex> lock(mtx_);
        pending_window_.assign(window.begin(), window.end());
        has_work_ = true;
    }
    busy_.store(true, std::memory_order_release);
    cv_.notify_one();
}


void DeepLearningBlock::Opaque::worker_loop()
{
    for (;;)
        {
            std::vector<gr_complex> window;
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait(lock, [this] { return has_work_ || stop_; });
                if (stop_ && !has_work_)
                    return;
                window = std::move(pending_window_);
                has_work_ = false;
            }

            try
                {
                    const auto t0 = std::chrono::steady_clock::now();
                    auto spectrogram = compute_spectrogram(window);
                    const auto t1 = std::chrono::steady_clock::now();
                    auto jammer = run_inference(spectrogram);
                    const auto t2 = std::chrono::steady_clock::now();

                    const double stft_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    const double onnx_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
                    const double total_ms = std::chrono::duration<double, std::milli>(t2 - t0).count();
                    const char *jammer_str = jammer_type_name(jammer);

                    std::cout << "Inference: " << jammer_str
                              << " (STFT " << stft_ms << " ms, ONNX " << onnx_ms << " ms)\n";

                    log_inference(jammer_str, stft_ms, onnx_ms, total_ms);
                    send_filter_command(jammer);
                }
            catch (const std::exception &e)
                {
                    std::cerr << "DeepLearningBlock: inference error: " << e.what() << "\n";
                }

            busy_.store(false, std::memory_order_release);
        }
}


DeepLearningBlock::sptr DeepLearningBlock::make(
    int window_size,
    const std::string &model_path,
    const std::string &log_dir,
    std::shared_ptr<Concurrent_Queue<pmt::pmt_t>> control_queue)
{
    return sptr(new DeepLearningBlock(window_size, model_path, log_dir, std::move(control_queue)));
}


DeepLearningBlock::DeepLearningBlock(
    int window_size,
    const std::string &model_path,
    const std::string &log_dir,
    std::shared_ptr<Concurrent_Queue<pmt::pmt_t>> control_queue)
    : gr::block("DeepLearningBlock",
          gr::io_signature::make(1, 1, sizeof(gr_complex)),
          gr::io_signature::make(0, 0, 0)),
      o_(new Opaque(this, window_size, model_path, log_dir, std::move(control_queue)))
{
}


DeepLearningBlock::~DeepLearningBlock()
{
    delete o_;
}


int DeepLearningBlock::general_work(int /*noutput_items*/,
    gr_vector_int &ninput_items,
    gr_vector_const_void_star &input_items,
    gr_vector_void_star & /*output_items*/)
{
    const auto *in = reinterpret_cast<const gr_complex *>(input_items[0]);
    const int n = ninput_items[0];

    for (int i = 0; i < n; ++i)
        {
            o_->buffer_.push_back(in[i]);
            if (static_cast<int>(o_->buffer_.size()) > o_->window_size_)
                o_->buffer_.pop_front();

            // Dispatch the freshest full window as soon as the worker is free --
            // no fixed interval to wait out. The worker's own processing time
            // (STFT+ONNX) is what naturally paces how often this actually
            // triggers; checking busy_ here (instead of only inside dispatch())
            // avoids calling dispatch() -- and its defensive drop-log -- on
            // every sample while the worker is still busy.
            if (static_cast<int>(o_->buffer_.size()) == o_->window_size_ &&
                !o_->busy_.load(std::memory_order_acquire))
                {
                    o_->dispatch(o_->buffer_);
                }
        }

    consume_each(n);
    return 0;
}
