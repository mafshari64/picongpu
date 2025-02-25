/* Copyright 2017-2024 Rene Widera
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
#include "picongpu/particles/filter/generic/Free.def"
#include "picongpu/particles/functor/User.hpp"

#include <string>


namespace picongpu
{
    namespace particles
    {
        namespace filter
        {
            namespace generic
            {
                namespace acc
                {
                    /** wrapper for the user filter on the accelerator
                     *
                     * @tparam T_Functor user defined filter
                     */
                    template<typename T_Functor>
                    struct Free : private T_Functor
                    {
                        //! type of the user filter
                        using Functor = T_Functor;

                        //! store user filter instance
                        HDINLINE Free(Functor const& filter) : Functor(filter)
                        {
                        }

                        /** execute the user filter
                         *
                         * @tparam T_Args type of the arguments passed to the user filter
                         *
                         * @param particle particle to use for the filtering
                         */
                        template<typename T_Worker, typename T_Particle>
                        HDINLINE bool operator()(T_Worker const&, T_Particle const& particle)
                        {
                            bool const isValid = particle.isHandleValid();

                            return isValid && Functor::operator()(particle);
                        }
                    };
                } // namespace acc

                template<typename T_Functor>
                struct Free : protected functor::User<T_Functor>
                {
                    using Functor = functor::User<T_Functor>;

                    template<typename T_SpeciesType>
                    struct apply
                    {
                        using type = Free;
                    };

                    /** constructor
                     *
                     * @param currentStep current simulation time step
                     */
                    HINLINE Free(uint32_t currentStep, IdGenerator idGen) : Functor(currentStep, idGen)
                    {
                    }

                    /** create device filter
                     *
                     * @tparam T_Worker lockstep worker type
                     *
                     * @param worker lockstep worker
                     * @param offset (in supercells, without any guards) to the
                     *         origin of the local domain
                     * @param configuration of the worker
                     */
                    template<typename T_Worker>
                    HDINLINE acc::Free<Functor> operator()(T_Worker const&, DataSpace<simDim> const&) const
                    {
                        return acc::Free<Functor>(*static_cast<Functor const*>(this));
                    }

                    HINLINE static std::string getName()
                    {
                        // provide the name from the user functor
                        return Functor::name;
                    }

                    /** A filter is deterministic if the filter outcome is equal between evaluations. If so, set this
                     * variable to true, otherwise to false.
                     *
                     * Example: A filter were results depend on a random number generator must return false.
                     */
                    static constexpr bool isDeterministic = Functor::isDeterministic;
                };

            } // namespace generic
        } // namespace filter
    } // namespace particles
} // namespace picongpu
