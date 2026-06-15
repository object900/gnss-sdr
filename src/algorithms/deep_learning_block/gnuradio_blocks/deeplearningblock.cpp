#include "deeplearningblock.h"
#include <vector>


class DeepLearningBlock::Opaque {
public:
    Opaque(DeepLearningBlock *parent) : p_(parent) {}

private:
    DeepLearningBlock *p_{nullptr};
    std::vector<gr_complex> buffer_;
    int window_size_;
    Ort::Session onnx_sessions_; // ONNX Runtime
    Ort::Env onnx_env_;

    int control_socket_; // port to GNSS-SDR filter block

    std::vector<float> compute_spectogram(const std::vector<gr_complex> &samples);
    int run_interference(const std::vector<float>& spectogram);
    void send_filter_command(JammerType jammer_type);
};

std::vector<float> Opaque::compute_spectogram(const std::vector<gr_complex> &samples) {

}

int Opaque::run_interference(const std::vector<float>& spectogram) {

}

void Opaque::send_filter_command(JammerType jammer_type) {

}

DeepLearningBlock::DeepLearningBlock() : o_(new Opaque(this)) {}

~DeepLearningBlock::DeepLearningBlock() {
    delete p_;
}

int DeepLearningBlock::general_work(int noutput_items,
                      gr_vector_int& ninput_items,
                      gr_vector_const_void_star& input_items,
                      gr_vector_void_star& output_items) {

    // get IQ
    // accumulate in buffer_
    // when buffer_ == window_size_
    // do STFT on buffer with FFTW
    // znormalizuj do formatu sieci (tensory, 512x512)
    // Ort::Session session(env, model_path, session_options);
    // auto output = session.Run(Ort::RunOptions{nullptr}, 
                            // input_names, &input_tensor, 1,
                            // output_names, 1);
    // najprostsze rozwiązanie to local socket (UNIX domain socket) lub shared memory między Twoim klasyfikatorem a zmodyfikowanym Input Filter blockiem.

    // Output port — blok musi mieć zdefiniowany output stream identyczny z input (przepuszcza I/Q dalej), bo flowgraf oczekuje że dane płyną do akwizycji i tracking.

    // MA DZIALAC W INNYM WATKU nieblokujac czytania - wymaga synchronizacji
    
}

