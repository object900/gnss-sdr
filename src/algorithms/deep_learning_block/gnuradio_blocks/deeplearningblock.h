#ifndef GNSS_SDR_DEEP_LEARNING_BLOCK_H
#define GNSS_SDR_DEEP_LEARNING_BLOCK_H

#include "command_event.h"
#include "concurrent_queue.h"
#include <gnuradio/block.h>
#include <pmt/pmt.h>
#include <memory>
#include <string>

// Kolejnosc MUSI byc dokladnie taka jak CLASSES w models/ResNet18.ipynb
enum class JammerType {
    CW, FM, CHIRP, PULSED, NARROWBAND, NOJAM
};

class DeepLearningBlock : public gr::block {
public:
    using sptr = std::shared_ptr<DeepLearningBlock>;

    // window_size: liczba probek IQ skladajacych sie na jedno okno analizy STFT
    //   (bufor przesuwny -- po zapelnieniu zawsze trzyma ostatnie window_size
    //   probek).
    // Brak osobnego "update_interval": okno jest dispatchowane do inferencji
    //   NATYCHMIAST gdy tylko watek roboczy jest wolny (patrz general_work()/
    //   dispatch() w .cpp) -- worker sam ogranicza tempo wlasnym czasem
    //   przetwarzania (STFT+ONNX), wiec sztywny throttle nie jest potrzebny do
    //   poprawnosci i tylko zwiekszalby opoznienie reakcji na zmiane sygnalu.
    // log_dir: katalog, w ktorym blok zapisuje spectrogram_live.pgm (podglad na
    //   zywo ostatniego okna STFT), inference_log.csv (werdykt + latencja STFT/ONNX
    //   dla kazdej inferencji) i filter_switch_log.csv (momenty przelaczania filtra).
    static sptr make(int window_size,
                     const std::string &model_path,
                     const std::string &log_dir,
                     std::shared_ptr<Concurrent_Queue<pmt::pmt_t>> control_queue);

    ~DeepLearningBlock();

    int general_work(int noutput_items,
                     gr_vector_int &ninput_items,
                     gr_vector_const_void_star &input_items,
                     gr_vector_void_star &output_items) override;

private:
    DeepLearningBlock(int window_size,
                      const std::string &model_path,
                      const std::string &log_dir,
                      std::shared_ptr<Concurrent_Queue<pmt::pmt_t>> control_queue);
    class Opaque;
    Opaque *o_;
};

#endif  // GNSS_SDR_DEEP_LEARNING_BLOCK_H
