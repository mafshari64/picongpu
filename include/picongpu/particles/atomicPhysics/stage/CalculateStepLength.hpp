/* Copyright 2023-2024 Brian Marre
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

/** @file findStepWidth sub-stage of atomicPhysics
 *
 * @attention assumes rate cache to be prefilled with diagonal elements of rate matrix
 *
 * find maximum possible atomicPhysics time step length for each superCell
 */

#pragma once

#include "picongpu/particles/atomicPhysics/kernel/CalculateStepLength.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/RateCacheField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeRemainingField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeStepField.hpp"
#include "picongpu/particles/param.hpp"

#include <pmacc/particles/meta/FindByNameOrType.hpp>

#include <string>

namespace picongpu::particles::atomicPhysics::stage
{
    /** @class atomic physics sub-stage for finding atomic physics step length
     *
     * @attention assumes timeStepField to have been reset before
     * @attention assumes rateCacheField to have been filled before
     *
     * @tparam T_IonSpecies ion species type
     */
    template<typename T_IonSpecies>
    struct CalculateStepLength
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_IonSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            auto& timeRemainingField = *dc.get<
                picongpu::particles::atomicPhysics::localHelperFields::TimeRemainingField<picongpu::MappingDesc>>(
                "TimeRemainingField");

            // pointers to memory, we will only work on device, no sync required
            //      pointer to rateCache
            auto& rateCacheField = *dc.get<picongpu::particles::atomicPhysics::localHelperFields::
                                               RateCacheField<picongpu::MappingDesc, IonSpecies>>(
                IonSpecies::FrameType::getName() + "_rateCacheField");
            //      pointer to timeStepField
            auto& timeStepField
                = *dc.get<picongpu::particles::atomicPhysics::localHelperFields::TimeStepField<picongpu::MappingDesc>>(
                    "TimeStepField");

            constexpr uint32_t numberAtomicStatesOfSpecies
                = picongpu::traits::GetNumberAtomicStates<IonSpecies>::value;

            // macro for call of kernel, see pull request #4321
            PMACC_LOCKSTEP_KERNEL(
                picongpu::particles::atomicPhysics::kernel::CalculateStepLengthKernel<numberAtomicStatesOfSpecies>())
                .template config<IonSpecies::FrameType::frameSize>(mapper.getGridDim())(
                    mapper,
                    timeRemainingField.getDeviceDataBox(),
                    timeStepField.getDeviceDataBox(),
                    rateCacheField.getDeviceDataBox());
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage
