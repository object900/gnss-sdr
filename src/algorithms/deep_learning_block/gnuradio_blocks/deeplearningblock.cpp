#include "deeplearningblock.h"
#include <gnuradio/io_signature.h>
#include <vector>

// command_id values for event_type=30 (set_input_filter), same as tcp_cmd_interface.cc
static constexpr int FILTER_CMD_PASS_THROUGH    = 301;
static constexpr int FILTER_CMD_NOTCH           = 302;
static constexpr int FILTER_CMD_PULSE_BLANKING  = 303;
static constexpr int FILTER_CMD_NOTCH_LITE      = 304;

class DeepLearningBlock::Opaque {
public:
    Opaque(DeepLearningBlock *parent,
           int window_size,
           std::shared_ptr<Concurrent_Queue<pmt::pmt_t>> queue)
        : p_(parent), window_size_(window_size), control_queue_(std::move(queue))
    {
        buffer_.reserve(window_size_);
    }

    int window_size_;
    std::vector<gr_complex> buffer_;
    std::shared_ptr<Concurrent_Queue<pmt::pmt_t>> control_queue_;

    // Ort::Env onnx_env_;
    // Ort::Session onnx_session_{nullptr};

    std::vector<float> compute_spectrogram(const std::vector<gr_complex> &samples);
    JammerType run_inference(const std::vector<float> &spectrogram);
    void send_filter_command(JammerType jammer_type);

private:
    DeepLearningBlock *p_{nullptr};
};


std::vector<float> DeepLearningBlock::Opaque::compute_spectrogram(
    const std::vector<gr_complex> &samples)
{
    // TODO: STFT z FFTW, normalizacja do 512x512
    return {};
}


JammerType DeepLearningBlock::Opaque::run_inference(
    const std::vector<float> &spectrogram)
{
    // TODO: Ort::Session::Run(...)
    return JammerType::NOJAM;
}


void DeepLearningBlock::Opaque::send_filter_command(JammerType jammer_type)
{
    if (!control_queue_)
        return;

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

    // identyczne z TcpCmdInterface::set_input_filter(), event_type=30
    const command_event_sptr new_evnt = command_event_make(command_id, 30);
    control_queue_->push(pmt::make_any(new_evnt));
}


DeepLearningBlock::sptr DeepLearningBlock::make(
    int window_size,
    std::shared_ptr<Concurrent_Queue<pmt::pmt_t>> control_queue)
{
    return sptr(new DeepLearningBlock(window_size, std::move(control_queue)));
}


DeepLearningBlock::DeepLearningBlock(
    int window_size,
    std::shared_ptr<Concurrent_Queue<pmt::pmt_t>> control_queue)
    : gr::block("DeepLearningBlock",
          gr::io_signature::make(1, 1, sizeof(gr_complex)),
          gr::io_signature::make(0, 0, 0)),
      o_(new Opaque(this, window_size, std::move(control_queue)))
{
}


DeepLearningBlock::~DeepLearningBlock()
{
    delete o_;
}


int DeepLearningBlock::general_work(int noutput_items,
    gr_vector_int &ninput_items,
    gr_vector_const_void_star &input_items,
    gr_vector_void_star & /*output_items*/)
{
    const auto *in = reinterpret_cast<const gr_complex *>(input_items[0]);
    const int n = ninput_items[0];

    for (int i = 0; i < n; ++i)
        {
            o_->buffer_.push_back(in[i]);

            if (static_cast<int>(o_->buffer_.size()) >= o_->window_size_)
                {
                    auto spectrogram = o_->compute_spectrogram(o_->buffer_);
                    auto jammer = o_->run_inference(spectrogram);
                    o_->send_filter_command(jammer);
                    o_->buffer_.clear();
                }
        }

    consume_each(n);
    return 0;
}
