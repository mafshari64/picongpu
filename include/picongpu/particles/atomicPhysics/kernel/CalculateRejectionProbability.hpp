/* Copyright 2024 Brian Marre
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

#include <cstdint>

namespace picongpu::particles::atomicPhysics::kernel
{
    /** methods for calculating the rejection probability
     *
     * @return sets the rejection probability cache entry for the specified entry to either
     *  -1, no rejection probability necessary
     *  >= 0, rejection probability
     *
     * also sets the flag sharedResourcesOverSubscribed to true if an over subscription is found
     */
    struct CalculateRejectionProbability
    {
        using VectorIdx = DataSpace<picongpu::simDim>;

        template<typename T_Histogram, typename T_RejectionProbabilityCache>
        HDINLINE static void ofHistogramBin(
            uint32_t const binIndex,
            T_Histogram const& histogram,
            T_RejectionProbabilityCache& rejectionProbabilityCache,
            bool& sharedResourcesOverSubscribed)
        {
            float_X const weight0 = histogram.getBinWeight0(binIndex);
            float_X const deltaWeight = histogram.getBinDeltaWeight(binIndex);

            float_X rejectionProbability = -1._X;

            if(weight0 < deltaWeight)
            {
                // bin is oversubscribed by suggested changes

                // calculate rejection probability
                rejectionProbability = math::max(
                    // proportion of weight we want to reject
                    (deltaWeight - weight0) / deltaWeight,
                    // but at least one average one macro ion should be rejected
                    picongpu::sim.unit.typicalNumParticlesPerMacroParticle() / deltaWeight);

                // set flag that we found at least one over subscribed resource
                sharedResourcesOverSubscribed = true;
            }

            rejectionProbabilityCache.(binIndex, rejectionProbability);
        }

        template<typename T_EFieldBox, typename T_EFieldEnergyUseCache, typename T_RejectionProbabilityCache>
        HDINLINE static void ofCell(
            uint32_t const linearCellIndex,
            VectorIdx const& superCellCellOffset,
            T_EFieldBox const eFieldBox,
            T_EFieldEnergyUseCache& eFieldEnergyUseCache,
            T_RejectionProbabilityCache& rejectionProbabilityCache,
            bool& sharedResourcesOverSubscribed)
        {
            VectorIdx const localCellIndex
                = pmacc::math::mapToND(picongpu::SuperCellSize::toRT(), static_cast<int>(linearCellIndex));
            VectorIdx const cellIndex = localCellIndex + superCellCellOffset;

            // sim.unit.charge()^2 * sim.unit.time()^2 / (sim.unit.mass() * sim.unit.length()^3)
            //  * sim.unit.length()^3
            // = sim.unit.charge()^2 * sim.unit.time()^2 / sim.unit.mass()
            constexpr float_X eps0HalfTimesCellVolume
                = (picongpu::sim.pic.getEps0() / 2._X) * picongpu::sim.pic.getCellSize().productOfComponents();

            // sim.unit.charge()^2 * sim.unit.time()^2 / sim.unit.mass()
            //  * ((sim.unit.mass() * sim.unit.length())/(sim.unit.time()^2 * sim.unit.charge()))^2
            // = sim.unit.charge()^2 * sim.unit.time()^2 * sim.unit.mass()^2 * sim.unit.length()^2
            //  / (sim.unit.mass() * sim.unit.time()^4 * sim.unit.charge()^2)
            // = sim.unit.mass() * sim.unit.length()^2/ (sim.unit.time()^2 * sim.unit.length())
            // sim.unit.energy()
            float_X const eFieldEnergy = eps0HalfTimesCellVolume * pmacc::math::l2norm2(eFieldBox(cellIndex));

            // eV * 1 = eV * sim.unit.energy()/sim.unit.energy() = (ev / sim.unit.energy()) * sim.unit.energy()
            // sim.unit.energy()
            float_X const eFieldEnergyUse
                = picongpu::sim.pic.get_eV() * eFieldEnergyUseCache.energyUsed(linearCellIndex);

            float_X rejectionProbability = -1._X;

            if(eFieldEnergyUse > eFieldEnergy)
            {
                // cell is oversubscribed by suggested changes

                // calculate rejection probability
                rejectionProbability = pmacc::math::max(
                    // proportion of weight we want to reject
                    (eFieldEnergyUse - eFieldEnergy) / eFieldEnergyUse,
                    // but approximately at least one average one macro ion per cell should be rejected
                    1._X / static_cast<float_X>(sim.getTypicalNumParticlesPerCell()));

                // set flag that we found at least one over subscribed resource
                sharedResourcesOverSubscribed = true;
            }

            rejectionProbabilityCache.setCell(linearCellIndex, rejectionProbability);
        }
    };
} // namespace picongpu::particles::atomicPhysics::kernel
