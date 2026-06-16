#include "deeplearningblock.h"
#include "gnss_sdr_fft.h"
#include "gnss_sdr_filesystem.h"
#include "onnx_model.h"
#include <gnuradio/io_signature.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
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
// Wymiary wejscia sieci - musi sie zgadzac z modelem .onnx (zob. modules/neural_network/src/main.cpp)
constexpr int kModelWidth = 512;
constexpr int kModelHeight = 512;

// Rozmiar pojedynczej ramki STFT i przesuniecie miedzy ramkami (50% nakladania)
constexpr int kStftFftSize = 256;
constexpr int kStftHop = kStftFftSize / 2;
constexpr float kPi = 3.14159265358979323846f;

const std::vector<float> &hamming_window()
{
    static const std::vector<float> w = [] {
        std::vector<float> win(kStftFftSize);
        for (int n = 0; n < kStftFftSize; ++n)
            win[n] = 0.54f - 0.46f * std::cos(2.0f * kPi * static_cast<float>(n) / (kStftFftSize - 1));
        return win;
    }();
    return w;
}

// Skaluje obraz src_w x src_h (row-major) do dst_w x dst_h interpolacja biliniowa,
// identycznie jak resize w modules/neural_network/src/image_loader.cpp
std::vector<float> bilinear_resize(const std::vector<float> &src,
    int src_w, int src_h, int dst_w, int dst_h)
{
    if (src_w == dst_w && src_h == dst_h)
        return src;

    std::vector<float> dst(static_cast<size_t>(dst_w) * dst_h);
    const float sx = static_cast<float>(src_w) / dst_w;
    const float sy = static_cast<float>(src_h) / dst_h;

    for (int oy = 0; oy < dst_h; ++oy)
        {
            float fy = (oy + 0.5f) * sy - 0.5f;
            int y0 = static_cast<int>(std::floor(fy));
            int y1 = y0 + 1;
            float dy = fy - y0;
            y0 = std::clamp(y0, 0, src_h - 1);
            y1 = std::clamp(y1, 0, src_h - 1);

            for (int ox = 0; ox < dst_w; ++ox)
                {
                    float fx = (ox + 0.5f) * sx - 0.5f;
                    int x0 = static_cast<int>(std::floor(fx));
                    int x1 = x0 + 1;
                    float dx = fx - x0;
                    x0 = std::clamp(x0, 0, src_w - 1);
                    x1 = std::clamp(x1, 0, src_w - 1);

                    float v = (1 - dy) * ((1 - dx) * src[y0 * src_w + x0] + dx * src[y0 * src_w + x1])
                            +      dy  * ((1 - dx) * src[y1 * src_w + x0] + dx * src[y1 * src_w + x1]);
                    dst[oy * dst_w + ox] = v;
                }
        }
    return dst;
}

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
        case JammerType::NARROW_BAND:
            return "Narrow Band";
        case JammerType::NOJAM:
            return "No Jam";
        case JammerType::SINGLE_AM:
            return "Single AM";
        case JammerType::SINGLE_FM:
            return "Single FM";
        case JammerType::SINGLE_CHIRP:
            return "Single Chirp";
        case JammerType::PULSED:
            return "DME (Pulsed)";
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
           int update_interval,
           const std::string &model_path,
           const std::string &log_dir,
           std::shared_ptr<Concurrent_Queue<pmt::pmt_t>> queue)
        : window_size_(window_size),
          update_interval_(update_interval),
          control_queue_(std::move(queue)),
          fft_(gnss_fft_fwd_make_unique(kStftFftSize)),
          p_(parent)
    {
        if (window_size_ < kStftFftSize)
            throw std::invalid_argument(
                "DeepLearningBlock: window_size must be >= " + std::to_string(kStftFftSize));

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
    int update_interval_;
    int samples_since_update_{0};
    std::deque<gr_complex> buffer_;
    std::shared_ptr<Concurrent_Queue<pmt::pmt_t>> control_queue_;
    std::unique_ptr<gnss_fft_complex_fwd> fft_;

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

    std::vector<float> compute_spectrogram(const std::vector<gr_complex> &samples);
    JammerType run_inference(const std::vector<float> &spectrogram);
    void send_filter_command(JammerType jammer_type);
    void dispatch(const std::deque<gr_complex> &window);

private:
    void worker_loop();
    void write_spectrogram_preview(const std::vector<float> &spectrogram);
    void log_inference(const char *jammer_str, double stft_ms, double onnx_ms, double total_ms);

    DeepLearningBlock *p_{nullptr};
    OnnxModel *om_{nullptr};
    std::thread worker_thread_;
};


std::vector<float> DeepLearningBlock::Opaque::compute_spectrogram(
    const std::vector<gr_complex> &samples)
{
    const int n_frames = (static_cast<int>(samples.size()) - kStftFftSize) / kStftHop + 1;
    const auto &window = hamming_window();

    // raw[freq_bin][frame], DC przesunieta do srodka (fftshift), row-major z szerokoscia n_frames
    std::vector<float> raw(static_cast<size_t>(kStftFftSize) * n_frames);

    for (int f = 0; f < n_frames; ++f)
        {
            const int offset = f * kStftHop;
            gr_complex *in = fft_->get_inbuf();
            for (int n = 0; n < kStftFftSize; ++n)
                in[n] = samples[offset + n] * window[n];

            fft_->execute();

            const gr_complex *out = fft_->get_outbuf();
            for (int k = 0; k < kStftFftSize; ++k)
                {
                    const int shifted = (k + kStftFftSize / 2) % kStftFftSize;
                    const float mag = std::abs(out[k]);
                    raw[static_cast<size_t>(shifted) * n_frames + f] = 20.0f * std::log10(mag + 1e-12f);
                }
        }

    const float min_db = *std::min_element(raw.begin(), raw.end());
    const float max_db = *std::max_element(raw.begin(), raw.end());
    const float range = std::max(max_db - min_db, 1e-6f);

    // Normalizacja do [0, 255], tak jak grayscale BMP w load_bmp_grayscale,
    // zeby skala wejscia odpowiadala temu, na czym siec byla trenowana.
    for (float &v : raw)
        v = (v - min_db) / range * 255.0f;

    auto resized = bilinear_resize(raw, n_frames, kStftFftSize, kModelWidth, kModelHeight);
    write_spectrogram_preview(resized);
    return resized;
}


void DeepLearningBlock::Opaque::write_spectrogram_preview(const std::vector<float> &spectrogram)
{
    // Nadpisywany plik PGM z najnowszym oknem STFT - do podgladu na zywo
    // (np. `feh --reload 1 spectrogram_live.pgm`). Pisanie do pliku tymczasowego
    // + rename jest atomowe, zeby viewer nigdy nie zlapal pol-zapisanego pliku.
    const std::string tmp_path = spectrogram_preview_path_ + ".tmp";
    std::ofstream f(tmp_path, std::ios::binary | std::ios::trunc);
    if (!f)
        return;

    f << "P5\n" << kModelWidth << " " << kModelHeight << "\n255\n";
    std::vector<uint8_t> pixels(spectrogram.size());
    for (size_t i = 0; i < spectrogram.size(); ++i)
        pixels[i] = static_cast<uint8_t>(std::clamp(spectrogram[i], 0.0f, 255.0f));
    f.write(reinterpret_cast<const char *>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
    f.close();

    errorlib::error_code ec;
    fs::rename(tmp_path, spectrogram_preview_path_, ec);
    if (ec)
        std::cerr << "DeepLearningBlock: could not update spectrogram preview: " << ec.message() << "\n";
}


JammerType DeepLearningBlock::Opaque::run_inference(
    const std::vector<float> &spectrogram)
{
    // Kolejnosc klas modelu jest alfabetyczna (zob. main.cpp): DME, NB, NoJam,
    // Single AM, Single Chirp, Single FM. DME ~ PULSED, NB ~ NARROW_BAND.
    static constexpr std::array<JammerType, 6> kClassToJammer = {
        JammerType::PULSED,
        JammerType::NARROW_BAND,
        JammerType::NOJAM,
        JammerType::SINGLE_AM,
        JammerType::SINGLE_CHIRP,
        JammerType::SINGLE_FM,
    };

    const std::vector<int64_t> input_shape = {1, 1, kModelHeight, kModelWidth};
    const auto output = om_->run(spectrogram, input_shape);

    const auto max_it = std::max_element(output.begin(), output.end());
    const auto class_index = static_cast<size_t>(std::distance(output.begin(), max_it));

    return (class_index < kClassToJammer.size()) ? kClassToJammer[class_index] : JammerType::NOJAM;
}


void DeepLearningBlock::Opaque::send_filter_command(JammerType jammer_type)
{
    int command_id = FILTER_CMD_PASS_THROUGH;
    switch (jammer_type)
        {
        case JammerType::SINGLE_AM:
        case JammerType::SINGLE_FM:
        case JammerType::NARROW_BAND:
            command_id = FILTER_CMD_NOTCH;
            break;
        case JammerType::PULSED:
            command_id = FILTER_CMD_PULSE_BLANKING;
            break;
        case JammerType::SINGLE_CHIRP:
            command_id = FILTER_CMD_NOTCH_LITE;
            break;
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
    if (busy_.load(std::memory_order_acquire))
        {
            std::cerr << "DeepLearningBlock: worker busy, dropping window\n";
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
    int update_interval,
    const std::string &model_path,
    const std::string &log_dir,
    std::shared_ptr<Concurrent_Queue<pmt::pmt_t>> control_queue)
{
    return sptr(new DeepLearningBlock(window_size, update_interval, model_path, log_dir, std::move(control_queue)));
}


DeepLearningBlock::DeepLearningBlock(
    int window_size,
    int update_interval,
    const std::string &model_path,
    const std::string &log_dir,
    std::shared_ptr<Concurrent_Queue<pmt::pmt_t>> control_queue)
    : gr::block("DeepLearningBlock",
          gr::io_signature::make(1, 1, sizeof(gr_complex)),
          gr::io_signature::make(0, 0, 0)),
      o_(new Opaque(this, window_size, update_interval, model_path, log_dir, std::move(control_queue)))
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

            ++o_->samples_since_update_;

            if (static_cast<int>(o_->buffer_.size()) == o_->window_size_ &&
                o_->samples_since_update_ >= o_->update_interval_)
                {
                    o_->samples_since_update_ = 0;
                    o_->dispatch(o_->buffer_);
                }
        }

    consume_each(n);
    return 0;
}
