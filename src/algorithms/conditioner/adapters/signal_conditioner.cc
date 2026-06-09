/*!
 * \file signal_conditioner.cc
 * \brief It holds blocks to change data type, filter and resample input data.
 * \author Luis Esteve, 2012. luis(at)epsilon-formacion.com
 *
 *
 * -----------------------------------------------------------------------------
 *
 * GNSS-SDR is a Global Navigation Satellite System software-defined receiver.
 * This file is part of GNSS-SDR.
 *
 * Copyright (C) 2010-2020  (see AUTHORS file for a list of contributors)
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * -----------------------------------------------------------------------------
 */

#include "signal_conditioner.h"
#include <stdexcept>
#include <utility>

#if USE_GLOG_AND_GFLAGS
#include <glog/logging.h>
#else
#include <absl/log/log.h>
#endif

// Constructor
SignalConditioner::SignalConditioner(std::shared_ptr<GNSSBlockInterface> data_type_adapt,
    std::shared_ptr<GNSSBlockInterface> in_filt,
    std::shared_ptr<GNSSBlockInterface> in_filt2,
    std::shared_ptr<GNSSBlockInterface> res,
    std::string role) : data_type_adapt_(std::move(data_type_adapt)),
                        in_filt_(std::move(in_filt)),
                        in_filt2_(std::move(in_filt2)),
                        res_(std::move(res)),
                        role_(std::move(role)),
                        connected_(false)
{
}


void SignalConditioner::connect(gr::top_block_sptr top_block)
{
    if (connected_)
        {
            LOG(WARNING) << "Signal conditioner already connected internally";
            return;
        }
    if (data_type_adapt_ == nullptr)
        {
            throw std::invalid_argument("DataTypeAdapter implementation not defined");
        }
    if (in_filt_ == nullptr)
        {
            throw std::invalid_argument("InputFilter implementation not defined");
        }
    if (res_ == nullptr)
        {
            throw std::invalid_argument("Resampler implementation not defined");
        }

    data_type_adapt_->connect(top_block);
    in_filt_->connect(top_block);
    if (in_filt2_ != nullptr)
        {
            in_filt2_->connect(top_block);
        }
    res_->connect(top_block);

    if (in_filt_->item_size() == 0)
        {
            throw std::invalid_argument("itemsize mismatch: Invalid input/output data type configuration for the InputFilter");
        }

    const size_t data_type_adapter_output_size = data_type_adapt_->get_right_block()->output_signature()->sizeof_stream_item(0);
    const size_t input_filter_input_size = in_filt_->get_left_block()->input_signature()->sizeof_stream_item(0);

    if (data_type_adapter_output_size != input_filter_input_size)
        {
            throw std::invalid_argument("itemsize mismatch: Invalid input/output data type configuration for the DataTypeAdapter/InputFilter connection");
        }

    top_block->connect(data_type_adapt_->get_right_block(), 0, in_filt_->get_left_block(), 0);
    DLOG(INFO) << "data_type_adapter -> input_filter";

    if (in_filt2_ != nullptr)
        {
            top_block->connect(in_filt_->get_right_block(), 0, in_filt2_->get_left_block(), 0);
            DLOG(INFO) << "input_filter -> input_filter2";
            top_block->connect(in_filt2_->get_right_block(), 0, res_->get_left_block(), 0);
            DLOG(INFO) << "input_filter2 -> resampler";
        }
    else
        {
            top_block->connect(in_filt_->get_right_block(), 0, res_->get_left_block(), 0);
            DLOG(INFO) << "input_filter -> resampler";
        }

    connected_ = true;
}


void SignalConditioner::disconnect(gr::top_block_sptr top_block)
{
    if (!connected_)
        {
            LOG(WARNING) << "Signal conditioner already disconnected internally";
            return;
        }

    top_block->disconnect(data_type_adapt_->get_right_block(), 0, in_filt_->get_left_block(), 0);

    if (in_filt2_ != nullptr)
        {
            top_block->disconnect(in_filt_->get_right_block(), 0, in_filt2_->get_left_block(), 0);
            top_block->disconnect(in_filt2_->get_right_block(), 0, res_->get_left_block(), 0);
        }
    else
        {
            top_block->disconnect(in_filt_->get_right_block(), 0, res_->get_left_block(), 0);
        }

    data_type_adapt_->disconnect(top_block);
    in_filt_->disconnect(top_block);
    if (in_filt2_ != nullptr)
        {
            in_filt2_->disconnect(top_block);
        }
    res_->disconnect(std::move(top_block));

    connected_ = false;
}


void SignalConditioner::switch_input_filter(std::shared_ptr<GNSSBlockInterface> new_input_filter, gr::top_block_sptr top_block)
{
    if (new_input_filter == nullptr)
        {
            throw std::invalid_argument("InputFilter implementation not defined");
        }

    if (in_filt_ != nullptr && in_filt_->implementation() == new_input_filter->implementation())
        {
            LOG(INFO) << "Signal conditioner input filter already set to " << new_input_filter->implementation();
            return;
        }

    if (!connected_)
        {
            in_filt_ = std::move(new_input_filter);
            LOG(INFO) << "Signal conditioner input filter switched to " << in_filt_->implementation();
            return;
        }

    auto old_input_filter = std::move(in_filt_);

    // Detach old filter, attach new one. in_filt2_ stays connected throughout.
    top_block->disconnect(data_type_adapt_->get_right_block(), 0, old_input_filter->get_left_block(), 0);
    if (in_filt2_ != nullptr)
        {
            top_block->disconnect(old_input_filter->get_right_block(), 0, in_filt2_->get_left_block(), 0);
        }
    else
        {
            top_block->disconnect(old_input_filter->get_right_block(), 0, res_->get_left_block(), 0);
        }
    old_input_filter->disconnect(top_block);

    new_input_filter->connect(top_block);
    top_block->connect(data_type_adapt_->get_right_block(), 0, new_input_filter->get_left_block(), 0);
    if (in_filt2_ != nullptr)
        {
            top_block->connect(new_input_filter->get_right_block(), 0, in_filt2_->get_left_block(), 0);
        }
    else
        {
            top_block->connect(new_input_filter->get_right_block(), 0, res_->get_left_block(), 0);
        }

    in_filt_ = std::move(new_input_filter);
    old_input_filter.reset();
    LOG(INFO) << "Signal conditioner input filter switched to " << in_filt_->implementation();
}


gr::basic_block_sptr SignalConditioner::get_left_block()
{
    return data_type_adapt_->get_left_block();
}


gr::basic_block_sptr SignalConditioner::get_right_block()
{
    return res_->get_right_block();
}
