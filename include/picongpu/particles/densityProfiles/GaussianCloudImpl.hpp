/* Copyright 2013-2024 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
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
#include "picongpu/simulation/control/MovingWindow.hpp"


namespace picongpu
{
    namespace densityProfiles
    {
        template<typename T_ParamClass>
        struct GaussianCloudImpl : public T_ParamClass
        {
            using ParamClass = T_ParamClass;

            template<typename T_SpeciesType>
            struct apply
            {
                using type = GaussianCloudImpl<ParamClass>;
            };

            HINLINE GaussianCloudImpl(uint32_t currentStep)
            {
            }

            /** Calculate the normalized density
             *
             * @param totalCellOffset total offset including all slides [in cells]
             */
            HDINLINE float_X operator()(const DataSpace<simDim>& totalCellOffset)
            {
                const float_64 unit_length = sim.unit.length();
                const float_X vacuum_y = float_X(ParamClass::vacuumCellsY) * sim.pic.getCellSize().y();
                constexpr auto centerSI = ParamClass::center_SI;
                const floatD_X center = precisionCast<float_X>(centerSI / unit_length);
                constexpr auto sigmaSI = ParamClass::sigma_SI;
                const floatD_X sigma = precisionCast<float_X>(sigmaSI / unit_length);

                const floatD_X globalCellPos(
                    precisionCast<float_X>(totalCellOffset) * sim.pic.getCellSize().shrink<simDim>());

                if(globalCellPos.y() < vacuum_y)
                    return float_X(0.0);

                /* for x, y, z calculate: x-x0 / sigma_x */
                const floatD_X r0overSigma = (globalCellPos - center) / sigma;
                /* get lenghts of r0 over sigma */
                const float_X exponent = pmacc::math::l2norm(r0overSigma);

                /* calculate exp(factor * exponent**power) */
                const float_X power = ParamClass::gasPower;
                const float_X factor = ParamClass::gasFactor;
                const float_X density = math::exp(factor * math::pow(exponent, power));

                return density;
            }
        };
    } // namespace densityProfiles
} // namespace picongpu
