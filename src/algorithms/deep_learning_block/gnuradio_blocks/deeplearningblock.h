#ifndef GNSS_SDR_DEEP_LEARNING_BLOCK_H
#define GNSS_SDR_DEEP_LEARNING_BLOCK_H

#include "command_event.h"
#include "concurrent_queue.h"
#include <gnuradio/block.h>
#include <pmt/pmt.h>
#include <memory>

enum class JammerType {
    NOJAM, SINGLE_AM, SINGLE_FM, SINGLE_CHIRP, PULSED, NARROW_BAND
};

class DeepLearningBlock : public gr::block {
public:
    using sptr = std::shared_ptr<DeepLearningBlock>;

    static sptr make(int window_size,
                     std::shared_ptr<Concurrent_Queue<pmt::pmt_t>> control_queue);

    ~DeepLearningBlock();

    int general_work(int noutput_items,
                     gr_vector_int &ninput_items,
                     gr_vector_const_void_star &input_items,
                     gr_vector_void_star &output_items) override;

private:
    DeepLearningBlock(int window_size,
                      std::shared_ptr<Concurrent_Queue<pmt::pmt_t>> control_queue);
    class Opaque;
    Opaque *o_;
};

#endif  // GNSS_SDR_DEEP_LEARNING_BLOCK_H
