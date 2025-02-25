/* Copyright 2016-2024 Marco Garten, Axel Huebl
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
#include "picongpu/fields/FieldTmpOperations.hpp"
#include "picongpu/fields/YeeCell.hpp"
#include "picongpu/particles/atomicPhysics/SetChargeState.hpp"
#include "picongpu/particles/ionization/byCollision/ThomasFermi/AlgorithmThomasFermi.hpp"
#include "picongpu/particles/ionization/byCollision/ThomasFermi/ThomasFermi.def"
#include "picongpu/traits/GetMargin.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/math/operation.hpp>
#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/memory/boxes/SharedBox.hpp>
#include <pmacc/meta/conversion/TypeToPointerPair.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>
#include <pmacc/random/RNGProvider.hpp>
#include <pmacc/random/distributions/Uniform.hpp>
#include <pmacc/random/methods/methods.hpp>
#include <pmacc/traits/Resolve.hpp>

namespace picongpu
{
    namespace particles
    {
        namespace ionization
        {
            /** ThomasFermi_Impl
             *
             * Thomas-Fermi pressure ionization - Implementation
             *
             * @tparam T_IonizationAlgorithm functor that returns a number of
             *         new free macro electrons to create, range: [0, boundElectrons]
             * @tparam T_DestSpecies type or name as PMACC_CSTRING of the electron species to be created
             * @tparam T_SrcSpecies type or name as PMACC_CSTRING of the particle species that is ionized
             */
            template<typename T_IonizationAlgorithm, typename T_DestSpecies, typename T_SrcSpecies>
            struct ThomasFermi_Impl
            {
                using DestSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_DestSpecies>;
                using SrcSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SrcSpecies>;

                using FrameType = typename SrcSpecies::FrameType;

                /** specify field to particle interpolation scheme
                 *
                 * @todo this needs to be done independently/twice if ion species (rho) and electron
                 *       species (ene) are of different shape
                 */
                using Field2ParticleInterpolation = typename pmacc::traits::Resolve<
                    typename pmacc::traits::GetFlagType<FrameType, interpolation<>>::type>::type;

                /* margins around the supercell for the interpolation of the field on the cells */
                using LowerMargin = typename picongpu::traits::GetMargin<Field2ParticleInterpolation>::LowerMargin;
                using UpperMargin = typename picongpu::traits::GetMargin<Field2ParticleInterpolation>::UpperMargin;

                /* relevant area of a block */
                using BlockArea = SuperCellDescription<typename MappingDesc::SuperCellSize, LowerMargin, UpperMargin>;

                BlockArea BlockDescription;

                /* parameter class containing the energy cutoff parameter for electron temperature calculation */
                struct CutoffMaxEnergy
                {
                    static constexpr float_X cutoffMaxEnergy = particles::ionization::thomasFermi::CUTOFF_MAX_ENERGY;
                };

            private:
                /* define ionization ALGORITHM (calculation) for ionization MODEL */
                using IonizationAlgorithm = T_IonizationAlgorithm;

                /* random number generator */
                using RNGFactory = pmacc::random::RNGProvider<simDim, random::Generator>;
                using Distribution = pmacc::random::distributions::Uniform<float_X>;
                using RandomGen = typename RNGFactory::GetRandomType<Distribution>::type;
                RandomGen randomGen;

                using SuperCellSize = MappingDesc::SuperCellSize;

                using ValueType_Rho = FieldTmp::ValueType;
                using ValueType_Ene = FieldTmp::ValueType;

                /* global memory EM-field device databoxes */
                PMACC_ALIGN(rhoBox, FieldTmp::DataBoxType);
                PMACC_ALIGN(eneBox, FieldTmp::DataBoxType);

                /* shared memory EM-field device databoxes */
                PMACC_ALIGN(
                    cachedRho,
                    pmacc::DataBox<pmacc::SharedBox<ValueType_Rho, typename BlockArea::FullSuperCellSize, 0>>);
                PMACC_ALIGN(
                    cachedEne,
                    pmacc::DataBox<pmacc::SharedBox<ValueType_Ene, typename BlockArea::FullSuperCellSize, 1>>);

            public:
                /* host constructor initializing member : random number generator */
                ThomasFermi_Impl(const uint32_t currentStep) : randomGen(RNGFactory::createRandom<Distribution>())
                {
                    /* create handle for access to host and device data */
                    DataConnector& dc = Environment<>::get().DataConnector();

                    /* The compiler is allowed to evaluate an expression that does not depend on a template parameter
                     * even if the class is never instantiated. In that case static assert is always
                     * evaluated (e.g. with clang), this results in an error if the condition is false.
                     * http://www.boost.org/doc/libs/1_60_0/doc/html/boost_staticassert.html
                     *
                     * A workaround is to add a template dependency to the expression.
                     * `sizeof(ANY_TYPE) != 0` is always true and defers the evaluation.
                     */
                    PMACC_CASSERT_MSG(
                        _please_allocate_at_least_two_FieldTmp_slots_in_memory_param,
                        (fieldTmpNumSlots >= 2) && (sizeof(T_IonizationAlgorithm) != 0));
                    /* initialize pointers on host-side density-/energy density field databoxes */
                    auto density = dc.get<FieldTmp>(FieldTmp::getUniqueId(0));
                    auto eneKinDens = dc.get<FieldTmp>(FieldTmp::getUniqueId(1));

                    /* reset density and kinetic energy values to zero */
                    density->getGridBuffer().getDeviceBuffer().setValue(FieldTmp::ValueType(0.));
                    eneKinDens->getGridBuffer().getDeviceBuffer().setValue(FieldTmp::ValueType(0.));

                    /* load species without copying the particle data to the host */
                    auto srcSpecies = dc.get<SrcSpecies>(SrcSpecies::FrameType::getName());

                    /** Calculate weighted ion density
                     *
                     * @todo Include all ion species because the model requires the
                     *       density of ionic potential wells
                     */
                    using DensitySolver = typename particleToGrid::
                        CreateFieldTmpOperation_t<SrcSpecies, particleToGrid::derivedAttributes::Density>::Solver;
                    computeFieldTmpValue<CORE + BORDER, DensitySolver>(*density, *srcSpecies, currentStep);

                    EventTask densityEvent = density->asyncCommunication(eventSystem::getTransactionEvent());
                    densityEvent += density->asyncCommunicationGather(densityEvent);

                    /* load species without copying the particle data to the host */
                    auto destSpecies = dc.get<DestSpecies>(DestSpecies::FrameType::getName());

                    /** Calculate energy density of the electron species with maximum energy cutoff
                     *
                     *  @todo Include all electron species with a meta::ForEach<VectorallSpecies,...>
                     * instead of just the destination species
                     */
                    using EnergyDensitySolver = typename particleToGrid::CreateFieldTmpOperation_t<
                        DestSpecies,
                        particleToGrid::derivedAttributes::EnergyDensityCutoff<CutoffMaxEnergy>>::Solver;
                    computeFieldTmpValue<CORE + BORDER, EnergyDensitySolver>(*eneKinDens, *destSpecies, currentStep);
                    EventTask eneKinEvent = eneKinDens->asyncCommunication(eventSystem::getTransactionEvent());
                    eneKinEvent += eneKinDens->asyncCommunicationGather(eneKinEvent);

                    /* contributions from neighboring GPUs to our border area */
                    eventSystem::setTransactionEvent(densityEvent + eneKinEvent);

                    /* initialize device-side density- and energy density field databox pointers */
                    rhoBox = density->getDeviceDataBox();
                    eneBox = eneKinDens->getDeviceDataBox();
                }

                /** cache fields used by this functor
                 *
                 * @warning this is a collective method and calls synchronize
                 *
                 * @tparam T_Worker lockstep worker type
                 *
                 * @param worker lockstep worker
                 * @param blockCell relative offset (in cells) to the local domain plus the guarding cells
                 * @param blockCfg configuration of the worker
                 */
                template<typename T_Worker>
                DINLINE void collectiveInit(const T_Worker& worker, const DataSpace<simDim>& blockCell)
                {
                    /* caching of density and "temperature" fields */
                    cachedRho = CachedBox::create<0, ValueType_Rho>(worker, BlockArea());
                    cachedEne = CachedBox::create<1, ValueType_Ene>(worker, BlockArea());

                    /* instance of nvidia assignment operator */
                    pmacc::math::operation::Assign assign;
                    /* copy fields from global to shared */
                    auto fieldRhoBlock = rhoBox.shift(blockCell);
                    auto collective = makeThreadCollective<BlockArea>();
                    collective(worker, assign, cachedRho, fieldRhoBlock);
                    /* copy fields from global to shared */
                    auto fieldEneBlock = eneBox.shift(blockCell);
                    collective(worker, assign, cachedEne, fieldEneBlock);

                    /* wait for shared memory to be initialized */
                    worker.sync();
                }

                /** Initialization function on device
                 *
                 * Cache density and energy density fields on device and initialize
                 * possible prerequisites for ionization, like e.g. random number
                 * generator.
                 *
                 * This function will be called inline on the device which must happen BEFORE threads diverge
                 * during loop execution. The reason for this is the `alpaka::syncBlockThreads( acc )` call which is
                 * necessary after initializing the field shared boxes in shared memory.
                 *
                 * @param localSuperCellOffset offset (in superCells, without any guards) relative
                 *                             to the origin of the local domain
                 * @param rngIdx linear index rng number index within the supercell, valid range[0;numFrameSlots)
                 */
                template<typename T_Worker>
                DINLINE void init(
                    [[maybe_unused]] T_Worker const& worker,
                    const DataSpace<simDim>& localSuperCellOffset,
                    const uint32_t rngIdx)
                {
                    auto rngOffset = DataSpace<simDim>::create(0);
                    rngOffset.x() = rngIdx;
                    auto numRNGsPerSuperCell = DataSpace<simDim>::create(1);
                    numRNGsPerSuperCell.x() = FrameType::frameSize;
                    this->randomGen.init(localSuperCellOffset * numRNGsPerSuperCell + rngOffset);
                }

                /** Determine number of new macro electrons due to ionization
                 *
                 * @param ionFrame reference to frame of the to-be-ionized particles
                 * @param localIdx local (linear) index in super cell / frame
                 */
                template<typename T_Worker>
                DINLINE uint32_t numNewParticles(T_Worker const& worker, FrameType& ionFrame, int localIdx)
                {
                    /* alias for the single macro-particle */
                    auto particle = ionFrame[localIdx];
                    /* particle position, used for field-to-particle interpolation */
                    floatD_X const pos = particle[position_];
                    int const particleCellIdx = particle[localCellIdx_];
                    /* multi-dim coordinate of the local cell inside the super cell */
                    DataSpace<SuperCellSize::dim> localCell
                        = pmacc::math::mapToND(SuperCellSize::toRT(), particleCellIdx);
                    /* interpolation of density */
                    const picongpu::traits::FieldPosition<fields::YeeCell, FieldTmp> fieldPosRho;
                    ValueType_Rho densityV
                        = Field2ParticleInterpolation()(cachedRho.shift(localCell), pos, fieldPosRho());
                    /*                          and energy density field on the particle position */
                    const picongpu::traits::FieldPosition<fields::YeeCell, FieldTmp> fieldPosEne;
                    ValueType_Ene kinEnergyV
                        = Field2ParticleInterpolation()(cachedEne.shift(localCell), pos, fieldPosEne());

                    /* density in sim units */
                    float_X const density = densityV[0];
                    /* energy density in sim units */
                    float_X const kinEnergyDensity = kinEnergyV[0];

                    /* Returns the new number of bound electrons for an integer number of macro electrons */
                    IonizationAlgorithm ionizeAlgo;
                    uint32_t newMacroElectrons
                        = ionizeAlgo(kinEnergyDensity, density, particle, this->randomGen(worker));


                    return newMacroElectrons;
                }

                /* Functor implementation
                 *
                 * Ionization model specific particle creation
                 *
                 * @tparam T_parentIon type of ion species that is being ionized
                 * @tparam T_childElectron type of electron species that is created
                 * @param parentIon ion instance that is ionized
                 * @param childElectron electron instance that is created
                 */
                template<typename T_parentIon, typename T_childElectron, typename T_Worker>
                DINLINE void operator()(
                    T_Worker const& worker,
                    IdGenerator& idGen,
                    T_parentIon& parentIon,
                    T_childElectron& childElectron)
                {
                    /* for not mixing operations::assign up with the nvidia functor assign */
                    namespace partOp = pmacc::particles::operations;
                    /* each thread sets the multiMask hard on "particle" (=1) */
                    childElectron[multiMask_] = 1u;
                    const float_X weighting = parentIon[weighting_];

                    /* each thread initializes a clone of the parent ion but leaving out
                     * some attributes:
                     * - multiMask: reading from global memory takes longer than just setting it again explicitly
                     * - momentum: because the electron would get a higher energy because of the ion mass
                     * - boundElectrons: because species other than ions or atoms do not have them
                     * (gets AUTOMATICALLY deselected because electrons do not have this attribute)
                     */
                    auto targetElectronClone = partOp::deselect<pmacc::mp_list<multiMask, momentum>>(childElectron);

                    targetElectronClone.derive(worker, idGen, parentIon);

                    const float_X massIon = picongpu::traits::attribute::getMass(weighting, parentIon);
                    const float_X massElectron = picongpu::traits::attribute::getMass(weighting, childElectron);

                    const float3_X electronMomentum(parentIon[momentum_] * (massElectron / massIon));

                    childElectron[momentum_] = electronMomentum;

                    /* conservation of momentum
                     * \todo add conservation of mass */
                    parentIon[momentum_] -= electronMomentum;

                    /** ionization of the ion by reducing the number of bound electrons
                     *
                     * @warning subtracting a float from a float can potentially
                     *          create a negative boundElectrons number for the ion,
                     *          see #1850 for details
                     */
                    float_X numberBoundElectrons = parentIon[boundElectrons_];

                    numberBoundElectrons -= 1._X;

                    picongpu::particles::atomicPhysics::SetChargeState{}(parentIon, numberBoundElectrons);
                }
            };

        } // namespace ionization
    } // namespace particles
} // namespace picongpu
