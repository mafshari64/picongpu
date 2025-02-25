/* Copyright 2024-2024 Brian Marre
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software you can redistribute it and or modify
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

#include <pmacc/static_assert.hpp>

namespace picongpu::particles::creation::moduleInterfaces
{
    /** interfaces of SanityCheckInputs
     *
     * interface for functor checking T_KernelConfigOptions, additionalData and source-/product-Boxes are consistent
     * with expectations and assumptions.
     *
     * @example check that:
     *   - if T_KernelConfigOptions specifies TransitionType as boundFree, checks that the transitionDataBox passed via
     *     additionalData actually contains boundFree transitions
     *   - the atomicNumbers of the chargeStateDataDataBox and atomicStateDataDataBox passed via additionalData are
     *     consistent
     */
    template<typename T_SourceParticleBox, typename T_ProductParticleBox, typename... T_KernelConfigOptions>
    struct SanityCheckInputs
    {
        //! @returns passes silently if okay
        template<typename T_Index, typename... T_AdditionalData>
        HDINLINE static void validate(
            pmacc::DataSpace<picongpu::simDim> const superCellIndex,
            T_Index const additionalDataIndex,
            T_AdditionalData... additionalData);
    };
} // namespace picongpu::particles::creation::moduleInterfaces
