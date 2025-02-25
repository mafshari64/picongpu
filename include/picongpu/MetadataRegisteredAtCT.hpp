/* Copyright 2024-2024 Julian Lenz
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/incidentField/param.hpp"
#include "picongpu/traits/GetMetadata.hpp"

#include <pmacc/meta/String.hpp>
#include <pmacc/meta/conversion/MakeSeq.hpp>

namespace picongpu
{
    using MetadataRegisteredAtCT = pmacc::MakeSeq_t<
        traits::IncidentFieldPolicy<PMACC_CSTRING("XMin"), picongpu::fields::incidentField::XMin>,
        traits::IncidentFieldPolicy<PMACC_CSTRING("XMax"), picongpu::fields::incidentField::XMax>,
        traits::IncidentFieldPolicy<PMACC_CSTRING("YMin"), picongpu::fields::incidentField::YMin>,
        traits::IncidentFieldPolicy<PMACC_CSTRING("YMax"), picongpu::fields::incidentField::YMax>,
        std::conditional_t<
            simDim == 2,
            pmacc::MakeSeq_t<>,
            pmacc::MakeSeq_t<
                traits::IncidentFieldPolicy<PMACC_CSTRING("ZMin"), picongpu::fields::incidentField::ZMin>,
                traits::IncidentFieldPolicy<PMACC_CSTRING("ZMax"), picongpu::fields::incidentField::ZMax>>>>;
} // namespace picongpu
