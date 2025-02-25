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

/** @file unit tests for the rate cross section calculation
 *
 * test are activated by the global debug switch debug::rateCalculation::RUN_UNIT_TESTS
 *  in atomicPhysics_Debug.param
 *
 * for updating the tests see the python [rate calculator tool](
 *  https://github.com/BrianMarre/picongpuAtomicPhysicsTools/tree/dev/RateCalculationReference)
 */

#pragma once

#include "picongpu/defines.hpp"
// need unit.param

#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"
#include "picongpu/particles/atomicPhysics/atomicData/AtomicTuples.def"
#include "picongpu/particles/atomicPhysics/enums/ADKLaserPolarization.hpp"
#include "picongpu/particles/atomicPhysics/enums/TransitionOrdering.hpp"
#include "picongpu/particles/atomicPhysics/rateCalculation/BoundBoundTransitionRates.hpp"
#include "picongpu/particles/atomicPhysics/rateCalculation/BoundFreeCollisionalTransitionRates.hpp"
#include "picongpu/particles/atomicPhysics/rateCalculation/BoundFreeFieldTransitionRates.hpp"
#include "picongpu/particles/atomicPhysics/stateRepresentation/ConfigNumber.hpp"

#include <pmacc/algorithms/math.hpp>

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <tuple>

namespace picongpu::particles::atomicPhysics::debug
{
    using tranOrd = picongpu::particles::atomicPhysics::enums::TransitionOrdering;

    /** collection of rate calculation tests
     *
     * see picongpuAtomicPhysicsTools repo, RateCalculationReference/calculatorMain.py
     *  for calculating reference rates.
     *
     * @tparam T_n_max maximum principal occupation number used in atomic state configNumber
     *      description
     * @tparam T_ConsoleOutput true =^= write result also to console
     */
    template<uint8_t T_n_max, bool T_consoleOutput = true>
    struct TestRateCalculation
    {
        static constexpr uint8_t numberLevels = 10u;
        static constexpr uint8_t atomicNumber = 4u;
        using ConfigNumberDataType = uint64_t;

        using S_ConfigNumber = picongpu::particles::atomicPhysics::stateRepresentation ::
            ConfigNumber<ConfigNumberDataType, numberLevels, atomicNumber>;

        static constexpr float_X energyElectron = 1000._X; // eV
        static constexpr float_X energyElectronBinWidth = 10._X; // eV
        static constexpr float_64 densityElectrons = 1.e28_X; // 1/(eV * m^3)

        using S_BoundBoundBuffer
            = atomicData::BoundBoundTransitionDataBuffer<uint32_t, float_X, uint32_t, uint64_t, tranOrd::byLowerState>;
        using S_BoundFreeBuffer = atomicData::BoundFreeTransitionDataBuffer<
            uint32_t,
            float_X,
            uint32_t,
            S_ConfigNumber,
            float_64,
            tranOrd::byLowerState>;
        using S_BoundBoundBox
            = atomicData::BoundBoundTransitionDataBox<uint32_t, float_X, uint32_t, uint64_t, tranOrd::byLowerState>;
        using S_BoundFreeBox = atomicData::
            BoundFreeTransitionDataBox<uint32_t, float_X, uint32_t, S_ConfigNumber, float_64, tranOrd::byLowerState>;

        using S_ChargeStateBuffer = atomicData::ChargeStateDataBuffer<uint32_t, float_X, S_ConfigNumber::atomicNumber>;
        using S_ChargeStateBox = atomicData::ChargeStateDataBox<uint32_t, float_X, S_ConfigNumber::atomicNumber>;

        using S_AtomicStateBuffer = atomicData::AtomicStateDataBuffer<uint32_t, float_X, S_ConfigNumber, float_64>;
        using S_AtomicStateBox = atomicData::AtomicStateDataBox<uint32_t, float_X, S_ConfigNumber, float_64>;

        std::unique_ptr<S_ChargeStateBuffer> chargeStateBuffer;
        std::unique_ptr<S_AtomicStateBuffer> atomicStateBuffer;
        std::unique_ptr<S_BoundBoundBuffer> boundBoundBuffer;
        std::unique_ptr<S_BoundFreeBuffer> boundFreeBuffer;


        TestRateCalculation()
        {
            // charge state already specifies number of entries
            chargeStateBuffer.reset(new S_ChargeStateBuffer());

            /// @attention number of states/transitions set for buffers needs to be == as actually added states in
            ///     setup()
            atomicStateBuffer.reset(new S_AtomicStateBuffer(5u));
            boundBoundBuffer.reset(new S_BoundBoundBuffer(1u));
            boundFreeBuffer.reset(new S_BoundFreeBuffer(2u));

            setup();
        }

    private:
        /** fill dataBuffers with small not physical example data
         *
         * dataBuffers are filled by hand to bypass checks of atomicData
         */
        void setup()
        {
            // charge states
            S_ChargeStateBox chargeStateHostBox = chargeStateBuffer->getHostDataBox();
            //      ionizationEnergy = 100 eV, screened charge = 5 e
            auto tupleChargeState_1 = std::make_tuple(u8(0u), 100._X, 5._X);
            auto tupleChargeState_2 = std::make_tuple(u8(1u), 5._X, 5._X);
            auto tupleChargeState_3 = std::make_tuple(u8(2u), 100._X, 5._X);

            chargeStateHostBox.store(u8(0u), tupleChargeState_1);
            chargeStateHostBox.store(u8(1u), tupleChargeState_2);
            chargeStateHostBox.store(u8(2u), tupleChargeState_3);

            chargeStateBuffer->hostToDevice();

            /// atomic states, @attention caution all atomic state must differ in configNumber
            S_AtomicStateBox atomicStateHostBox = atomicStateBuffer->getHostDataBox();

            // 1:(1,1,0,0,0,0,1,0,1,0) lowerStateBoundFree
            auto tupleAtomicState_bf_1 = std::make_tuple(static_cast<uint64_t>(243754u), 0._X);
            // 2:(1,1,0,0,0,0,1,0,0,0) upperStateBoundFree, excitationEnergyDifference = 5 eV
            auto tupleAtomicState_bf_2 = std::make_tuple(static_cast<uint64_t>(9379u), 5._X);
            // 3:(1,1,0,0,0,0,0,0,0,0) upperStateBoundFree, excitationEnergyDifference = 5 eV
            auto tupleAtomicState_bf_3 = std::make_tuple(static_cast<uint64_t>(4u), 5._X);

            /// @note states must be sorted primarily ascending by charge state, secondarily ascending by configNumber
            atomicStateHostBox.store(u8(1u), tupleAtomicState_bf_1);
            atomicStateHostBox.store(u8(3u), tupleAtomicState_bf_2);
            atomicStateHostBox.store(u8(4u), tupleAtomicState_bf_3);

            // 1:(1,0,2,0,0,0,1,0,0,0) lowerStateBoundBound
            auto tupleAtomicState_bb_1 = std::make_tuple(static_cast<uint64_t>(9406u), 0._X);
            // 1:(1,0,1,0,0,0,1,0,1,0) upperStateBoundBound, energyDiffLowerUpper = 5 eV
            auto tupleAtomicState_bb_2 = std::make_tuple(static_cast<uint64_t>(243766u), 5._X);

            /// @note states must be sorted primarily ascending by charge state, secondarily ascending by configNumber
            atomicStateHostBox.store(u8(0u), tupleAtomicState_bb_1);
            atomicStateHostBox.store(u8(2u), tupleAtomicState_bb_2);

            atomicStateBuffer->hostToDevice();

            // bound-bound transitions
            S_BoundBoundBox boundBoundHostBox = boundBoundBuffer->getHostDataBox();
            /* collisionalOscillatorStrength = 1, absorptionOscillatorStrength = 1e-1,
             *  cxin1 = 1., cxin2 = 2., cxin3 = 3., cxin4 = 4.cxin5 = 5.
             */
            auto tupleBoundBound = std::make_tuple(
                1._X,
                0.1_X,
                1._X,
                2._X,
                3._X,
                4._X,
                5._X,
                static_cast<uint64_t>(std::get<0>(tupleAtomicState_bb_1)),
                static_cast<uint64_t>(std::get<0>(tupleAtomicState_bb_2)));
            boundBoundHostBox.store(0u, tupleBoundBound, atomicStateHostBox);
            boundBoundBuffer->hostToDevice();

            // bound-free transition
            auto tupleBoundFree_1 = std::make_tuple(
                1._X,
                2._X,
                3._X,
                4._X,
                5._X,
                6._X,
                7._X,
                8._X,
                static_cast<uint64_t>(std::get<0>(tupleAtomicState_bf_1)),
                static_cast<uint64_t>(std::get<0>(tupleAtomicState_bf_2)));
            auto tupleBoundFree_2 = std::make_tuple(
                1._X,
                2._X,
                3._X,
                4._X,
                5._X,
                6._X,
                7._X,
                8._X,
                static_cast<uint64_t>(std::get<0>(tupleAtomicState_bf_2)),
                static_cast<uint64_t>(std::get<0>(tupleAtomicState_bf_3)));

            S_BoundFreeBox boundFreeHostBox = boundFreeBuffer->getHostDataBox();
            boundFreeHostBox.store(0u, tupleBoundFree_1, atomicStateHostBox);
            boundFreeHostBox.store(1u, tupleBoundFree_2, atomicStateHostBox);
            boundFreeBuffer->hostToDevice();
        }

        /** test for relative error limit
         *
         * @tparam T_Type data type of quantity
         * @tparam T_consoleOutput true =^= write output also to console, false =^= no console output
         *
         * @param correctValue expected value
         * @param testValue actual value
         * @param descriptionQuantity short description of quantity tested
         * @param errorLimit maximm accepted relative error
         *
         * @attention may only be executed serially on cpu
         *
         * @return true =^= SUCCESS, false =^= FAIL
         */
        template<typename T_Type>
        static bool testRelativeError(
            T_Type const correctValue,
            T_Type const testValue,
            std::string const descriptionQuantity = "",
            T_Type const errorLimit = static_cast<T_Type>(1e-9))
        {
            T_Type const relativeError = math::abs(testValue / correctValue - static_cast<T_Type>(1.f));

            if constexpr(T_consoleOutput)
            {
                std::cout << "[relative error]" << descriptionQuantity << ":\t" << relativeError << std::endl;
            }

            if(std::isnan(relativeError))
            {
                // FAIL
                if constexpr(T_consoleOutput)
                    std::cout << "\t x FAIL, is NaN" << std::endl;
                return false;
            }

            if(relativeError > errorLimit)
            {
                // FAIL
                if constexpr(T_consoleOutput)
                    std::cout << "\t x FAIL, > errorLimit" << std::endl;
                return false;
            }
            else
            {
                // SUCCESS
                if constexpr(T_consoleOutput)
                    std::cout << "\t * SUCCESS" << std::endl;
                return true;
            }
        }

    public:
        //! @return true =^= test passed
        bool testCollisionalExcitationCrossSection() const
        {
            float_X const correctCrossSection = 3.456217425189e+02; // 1e6b
            float_X const crossSection = rateCalculation::BoundBoundTransitionRates<T_n_max>::
                template collisionalBoundBoundCrossSection<S_AtomicStateBox, S_BoundBoundBox, true>(
                    energyElectron,
                    u32(0u),
                    atomicStateBuffer->getHostDataBox(),
                    boundBoundBuffer->getHostDataBox()); // 1e6b

            return testRelativeError(
                correctCrossSection,
                crossSection,
                "collisional excitation cross section",
                static_cast<float_X>(1e-5));
        }

        //! @return true =^= test passed
        bool testCollisionalDeexcitationCrossSection() const
        {
            float_X const energyElectron = 1000._X;
            float_X const correctCrossSection = 1.814666351842e+01; // 1e6b
            float_X const crossSection = rateCalculation::BoundBoundTransitionRates<T_n_max>::
                template collisionalBoundBoundCrossSection<S_AtomicStateBox, S_BoundBoundBox, false>(
                    energyElectron,
                    0u,
                    atomicStateBuffer->getHostDataBox(),
                    boundBoundBuffer->getHostDataBox()); // 1e6

            return testRelativeError(
                correctCrossSection,
                crossSection,
                "collisional deexcitation cross section",
                static_cast<float_X>(1.e-5));
        }

        //! @return true =^= test passed
        bool testCollisionalIonizationCrossSection() const
        {
            float_X const correctCrossSection = 8.051678880120e-01; // 1e6b
            float_X const crossSection = rateCalculation::BoundFreeCollisionalTransitionRates<T_n_max, true>::
                collisionalIonizationCrossSection(
                    // eV
                    energyElectron,
                    // ionization potential depression, eV
                    0._X,
                    0u,
                    chargeStateBuffer->getHostDataBox(),
                    atomicStateBuffer->getHostDataBox(),
                    boundFreeBuffer->getHostDataBox()); // 1e6b

            return testRelativeError(
                correctCrossSection,
                crossSection,
                "collisional ionization cross section",
                static_cast<float_X>(
                    1e-3)); /// @todo find out why error is larger than for de-/excitation, Brian Marre, 2023
        }

        //! @return true =^= test passed
        bool testCollisionalExcitationRate() const
        {
            float_64 const correctRate = 6.472768268762e+16; // 1/s
            float_64 const rate
                = static_cast<float_64>(
                      rateCalculation::BoundBoundTransitionRates<T_n_max>::
                          template rateCollisionalBoundBoundTransition<S_AtomicStateBox, S_BoundBoundBox, true>(
                              energyElectron,
                              energyElectronBinWidth,
                              // 1/(eV*m^3) * (m/sim.unit.length())^3 = = 1/(eV * sim.unit.length()^3)
                              static_cast<float_X>(
                                  densityElectrons * pmacc::math::cPow(picongpu::sim.unit.length(), 3u)),
                              0u,
                              atomicStateBuffer->getHostDataBox(),
                              boundBoundBuffer->getHostDataBox()))
                * 1. / sim.unit.time(); // 1/s

            return testRelativeError(correctRate, rate, "collisional excitation rate", 1e-5);
        }

        //! @return true =^= test passed
        bool testCollisionalDeexcitationRate() const
        {
            float_64 const correctRate = 3.398488386461e+15; // 1/s
            float_64 const rate
                = static_cast<float_64>(
                      rateCalculation::BoundBoundTransitionRates<T_n_max>::
                          template rateCollisionalBoundBoundTransition<S_AtomicStateBox, S_BoundBoundBox, false>(
                              energyElectron,
                              energyElectronBinWidth,
                              static_cast<float_X>(
                                  densityElectrons * pmacc::math::cPow(picongpu::sim.unit.length(), 3u)),
                              0u,
                              atomicStateBuffer->getHostDataBox(),
                              boundBoundBuffer->getHostDataBox()))
                * 1. / sim.unit.time(); // 1/s

            return testRelativeError(correctRate, rate, "collisional deexcitation rate", 1e-5);
        }

        //! @return true =^= test passed
        bool testSpontaneousRadiativeDeexcitationRate() const
        {
            float_64 const correctRate = 5.691850311676e+06; // 1/s
            float_64 const rate
                = static_cast<float_64>(
                      rateCalculation::BoundBoundTransitionRates<T_n_max>::rateSpontaneousRadiativeDeexcitation(
                          0u,
                          atomicStateBuffer->getHostDataBox(),
                          boundBoundBuffer->getHostDataBox()))
                * 1. / sim.unit.time(); // 1/s

            return testRelativeError(correctRate, rate, "spontaneous radiative deexcitation rate", 1e-5);
        }

        //! @return true =^= test passed, pass silently if correct
        bool testCollisionalIonizationRate() const
        {
            float_64 const correctRate = 1.507910098065e+14; // 1/s
            float_64 const rate
                = static_cast<float_64>(
                      rateCalculation::BoundFreeCollisionalTransitionRates<T_n_max, true>::
                          rateCollisionalIonizationTransition(
                              energyElectron,
                              energyElectronBinWidth,
                              static_cast<float_X>(
                                  densityElectrons * pmacc::math::cPow(picongpu::sim.unit.length(), 3u)),
                              // ionization potential depression
                              0._X,
                              0u,
                              chargeStateBuffer->getHostDataBox(),
                              atomicStateBuffer->getHostDataBox(),
                              boundFreeBuffer->getHostDataBox()))
                * 1. / sim.unit.time(); // 1/s

            //! @note larger error limit required due to numerics of rate formula
            return testRelativeError(correctRate, rate, "collisional ionization rate", 1e-3);
        }

        //! @return true =^= test passed
        bool testADKIonizationRate() const
        {
            // unit: 1/s
            float_64 const correctRate = 6.391666527e+9 * 1 / 3.3e-17;

            // unit: unit_eField
            float_X const eFieldNorm = 0.0126 * sim.atomicUnit.eField() / sim.unit.eField();

            // unit: eV
            float_X const ipd = 0._X;

            constexpr auto laserPolarization
                = picongpu::particles::atomicPhysics::enums::ADKLaserPolarization::linearPolarization;
            // unit: 1/s
            float_64 const rate
                = static_cast<float_64>(
                      rateCalculation::BoundFreeFieldTransitionRates<laserPolarization>::rateADKFieldIonization<
                          float_X>(
                          eFieldNorm,
                          ipd,
                          u32(1u),
                          chargeStateBuffer->getHostDataBox(),
                          atomicStateBuffer->getHostDataBox(),
                          boundFreeBuffer->getHostDataBox()))
                * 1. / sim.unit.time();

            /// @note larger error limit required due to numerics of ADK rate formula
            return testRelativeError(correctRate, rate, "ADK field ionization", 1e-5);
        }

        //! @return true =^= all tests passed
        bool testAll()
        {
            constexpr uint8_t numberTests = 8;
            bool pass[numberTests];
            pass[0] = testCollisionalExcitationCrossSection();
            pass[1] = testCollisionalDeexcitationCrossSection();
            pass[2] = testCollisionalIonizationCrossSection();
            pass[3] = testCollisionalExcitationRate();
            pass[4] = testCollisionalDeexcitationRate();
            pass[5] = testSpontaneousRadiativeDeexcitationRate();
            pass[6] = testCollisionalIonizationRate();
            pass[7] = testADKIonizationRate();

            bool pass_total = true;
            for(uint8_t i = 0u; i < numberTests; ++i)
            {
                pass_total = pass_total && pass[i];
            }

            if constexpr(T_consoleOutput)
            {
                std::cout << "Result:";
                if(pass_total)
                    std::cout << " * Success" << std::endl;
                else
                    std::cout << " x Fail" << std::endl;
            }
            return pass_total;
        }
    };
} // namespace picongpu::particles::atomicPhysics::debug
