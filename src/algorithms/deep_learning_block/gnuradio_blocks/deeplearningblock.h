#include <gnuradio/block.h>

enum class JammerType {
    NOJAM, SINGLE_AM, SINGLE_FM, SINGLE_CHIRP, PULSED, NARROW_BAND
};

class DeepLearningBlock : public gr::block {
public:
    DeepLearningBlock() = default;
    ~DeepLearningBlock() = default;
    int general_work(int noutput_items,
                      gr_vector_int& ninput_items,
                      gr_vector_const_void_star& input_items,
                      gr_vector_void_star& output_items) override;
private:
    class Opaque;
    Opaque *o_;
};