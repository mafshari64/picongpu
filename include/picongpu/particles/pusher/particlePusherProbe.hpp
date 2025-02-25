/* Copyright 2017-2024 Axel Huebl
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

#include <pmacc/math/operation.hpp>
#include <pmacc/particles/operations/Deselect.hpp>
#include <pmacc/traits/HasIdentifier.hpp>

namespace picongpu
{
    namespace particlePusherProbe
    {
        /** Probe electro-magnetic fields and store the result with a particle
         *
         * @tparam T_ValueFunctor pmacc::math::operation::*, binary functor
         *         handling how to store the obtained field on the particle,
         *         default is assigning a new value
         * @tparam T_ActualPush allows to perform a real particle push after
         *         probing the electro-magnetic field (e.g. to let a probe
         *         particle stream with a moving window or to define a tracer
         *         particle species that records its fields),
         *         default is void and means no push (just a static probe)
         */
        template<typename T_ValueFunctor = pmacc::math::operation::Assign, typename T_ActualPush = void>
        struct Push
        {
            using ActualPush = T_ActualPush;

            /* this is an optional extension for sub-sampling pushes that enables grid to particle interpolation
             * for particle positions outside the super cell in one push
             */
            using LowerMargin = typename ActualPush::LowerMargin;
            using UpperMargin = typename ActualPush::UpperMargin;

            template<typename T_FunctorFieldE, typename T_FunctorFieldB, typename T_Particle, typename T_Pos>
            HDINLINE void operator()(
                T_FunctorFieldB const functorBField,
                T_FunctorFieldE const functorEField,
                T_Particle& particle,
                T_Pos& pos,
                uint32_t const currentStep)
            {
                T_ValueFunctor valueFunctor;
                const auto bField = functorBField(pos);
                const auto eField = functorEField(pos);

                // update probe field if particle contains required attributes
                if constexpr(pmacc::traits::HasIdentifier<T_Particle, probeB>::type::value)
                    valueFunctor(particle[probeB_], bField);
                if constexpr(pmacc::traits::HasIdentifier<T_Particle, probeE>::type::value)
                    valueFunctor(particle[probeE_], eField);

                ActualPush actualPush;
                // remove probe attribute from the particle, this data is already recorded and should not be recorded
                // by actualPush again
                auto filteredParticle = pmacc::particles::operations::deselect<MakeSeq_t<probeB, probeE>>(particle);
                actualPush(functorBField, functorEField, filteredParticle, pos, currentStep);
            }

            static pmacc::traits::StringProperty getStringProperties()
            {
                pmacc::traits::GetStringProperties<ActualPush> propList;
                propList["param"] = "moving probe";
                return propList;
            }
        };

        template<typename T_ValueFunctor>
        struct Push<T_ValueFunctor, void>
        {
            /* this is an optional extension for sub-sampling pushes that enables grid to particle interpolation
             * for particle positions outside the super cell in one push
             */
            using LowerMargin = typename pmacc::math::CT::make_Int<simDim, 0>::type;
            using UpperMargin = typename pmacc::math::CT::make_Int<simDim, 0>::type;

            template<typename T_FunctorFieldE, typename T_FunctorFieldB, typename T_Particle, typename T_Pos>
            HDINLINE void operator()(
                T_FunctorFieldB const functorBField,
                T_FunctorFieldE const functorEField,
                T_Particle& particle,
                T_Pos& pos,
                uint32_t const)
            {
                T_ValueFunctor valueFunctor;
                const auto bField = functorBField(pos);
                if constexpr(pmacc::traits::HasIdentifier<T_Particle, probeB>::type::value)
                    valueFunctor(particle[probeB_], bField);
                const auto eField = functorEField(pos);
                if constexpr(pmacc::traits::HasIdentifier<T_Particle, probeE>::type::value)
                    valueFunctor(particle[probeE_], eField);
            }

            static pmacc::traits::StringProperty getStringProperties()
            {
                pmacc::traits::StringProperty propList("name", "other");
                propList["param"] = "static probe";
                return propList;
            }
        };
    } // namespace particlePusherProbe
} // namespace picongpu
