/* Copyright 2016-2024 Alexander Grund
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <pmacc/boost_workaround.hpp>

#include <pmacc/Environment.hpp>
#include <pmacc/assert.hpp>
#include <pmacc/dataManagement/ISimulationData.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/eventSystem/tasks/ITask.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/random/RNGProvider.hpp>
#include <pmacc/random/distributions/Uniform.hpp>
#include <pmacc/random/methods/AlpakaRand.hpp>
#include <pmacc/simulationControl/TimeInterval.hpp>
#include <pmacc/types.hpp>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>


namespace pmacc
{
    namespace test
    {
        namespace random
        {
            using Space2D = pmacc::DataSpace<DIM2>;
            using Space3D = pmacc::DataSpace<DIM3>;

            template<uint32_t T_blockSize>
            struct RandomFiller
            {
                template<typename T_DataBox, typename T_Random, typename T_Worker>
                DINLINE void operator()(
                    T_Worker const& worker,
                    T_DataBox box,
                    Space2D const boxSize,
                    T_Random const rand,
                    uint32_t const numSamples) const
                {
                    // each virtual worker initialize one rng state
                    auto forEachCell = lockstep::makeForEach<T_blockSize>(worker);

                    forEachCell(
                        [&](int32_t const linearIdx)
                        {
                            int32_t const linearTid = worker.blockDomIdxND().x() * T_blockSize + linearIdx;

                            if(linearTid >= boxSize.productOfComponents())
                                return;

                            Space2D const ownIdx = pmacc::math::mapToND(boxSize, linearTid);
                            // each virtual worker needs an own instance of rand
                            T_Random vWorkerRand = rand;
                            vWorkerRand.init(ownIdx);
                            for(uint32_t i = 0u; i < numSamples; i++)
                            {
                                Space2D idx = vWorkerRand(worker, boxSize);
                                alpaka::atomicAdd(worker.getAcc(), &box(idx), 1u, ::alpaka::hierarchy::Blocks{});
                            }
                        });
                }
            };

            template<class T_RNGProvider>
            struct GetRanidx
            {
                typedef pmacc::random::distributions::Uniform<float> Distribution;
                typedef typename T_RNGProvider::template GetRandomType<Distribution>::type Random;

                HINLINE GetRanidx() : rand(T_RNGProvider::template createRandom<Distribution>())
                {
                }

                /** initialize the random generator
                 *
                 * @warning: it is not allowed to call this method twice on an instance
                 */
                DINLINE void init(Space2D globalCellIdx)
                {
                    rand.init(globalCellIdx);
                }

                template<typename T_Worker>
                DINLINE Space2D operator()(T_Worker const& worker, Space2D size)
                {
                    using pmacc::math::float2int_rd;
                    return Space2D(float2int_rd(rand(worker) * size.x()), float2int_rd(rand(worker) * size.y()));
                }

            private:
                Random rand;
            };

            /** Write in PGM grayscale file format (easy to read/interpret) */
            template<class T_Buffer>
            void writePGM(const std::string& filePath, T_Buffer& buffer)
            {
                const Space2D size = buffer.capacityND();
                uint32_t maxVal = 0;
                for(int y = 0; y < size.y(); y++)
                {
                    for(int x = 0; x < size.x(); x++)
                    {
                        uint32_t val = buffer.getDataBox()(Space2D(x, y));
                        if(val > maxVal)
                            maxVal = val;
                    }
                }

                // Standard format is single byte per value which limits the range to 0-255
                // An extension allows 2 bytes so 0-65536)
                if(maxVal > std::numeric_limits<uint16_t>::max())
                    maxVal = std::numeric_limits<uint16_t>::max();
                const bool isTwoByteFormat = maxVal > std::numeric_limits<uint8_t>::max();

                std::ofstream outFile(filePath.c_str());
                // TAG
                outFile << "P5\n";
                // Size and maximum value (at most 65536 which is 2 bytes per value)
                outFile << size.x() << " " << size.y() << " " << maxVal << "\n";
                for(int y = 0; y < size.y(); y++)
                {
                    for(int x = 0; x < size.x(); x++)
                    {
                        uint32_t val = buffer.getDataBox()(Space2D(x, y));
                        // Clip value
                        if(val > maxVal)
                            val = maxVal;
                        // Write first byte (higher order bits) if file is in 2 byte format
                        if(isTwoByteFormat)
                            outFile << uint8_t(val >> 8);
                        // Write remaining bytze
                        outFile << uint8_t(val);
                    }
                }
            }

            template<class T_DeviceBuffer, class T_Random>
            void generateRandomNumbers(
                const Space2D& rngSize,
                uint32_t numSamples,
                T_DeviceBuffer& buffer,
                const T_Random& rand)
            {
                pmacc::TimeIntervall timer;
                timer.toggleStart();

                constexpr uint32_t blockSize = 256;

                uint32_t gridSize = (rngSize.productOfComponents() + blockSize - 1u) / blockSize;

                eventSystem::getTransactionEvent().waitForFinished();
                timer.toggleStart();

                PMACC_LOCKSTEP_KERNEL(RandomFiller<blockSize>{})
                    .template config<blockSize>(gridSize)(buffer.getDataBox(), buffer.capacityND(), rand, numSamples);

                eventSystem::getTransactionEvent().waitForFinished();
                timer.toggleEnd();

                float milliseconds = timer.getInterval();
                std::cout << "Done in " << milliseconds << "ms" << std::endl;
            }

            template<class T_Method>
            void runTest(uint32_t numSamples)
            {
                typedef pmacc::random::RNGProvider<2, T_Method> RNGProvider;

                const std::string rngName = RNGProvider::RNGMethod::getName();
                std::cout << std::endl
                          << "Running test for " << rngName << " with " << numSamples << " samples per cell"
                          << std::endl;
                // Size of the detector
                const Space2D size(256, 256);
                // Size of the rng provider (= number of states used)
                const Space2D rngSize(256, 256);

                pmacc::HostDeviceBuffer<uint32_t, 2> detector(size);
                auto rngProvider = new RNGProvider(rngSize);

                pmacc::Environment<>::get().DataConnector().share(
                    std::shared_ptr<pmacc::ISimulationData>(rngProvider));
                rngProvider->init(0x42133742);

                generateRandomNumbers(rngSize, numSamples, detector.getDeviceBuffer(), GetRanidx<RNGProvider>());

                detector.deviceToHost();
                auto box = detector.getHostBuffer().getDataBox();
                // Write data to file
                std::ofstream dataFile((rngName + "_data.txt").c_str());
                for(int y = 0; y < size.y(); y++)
                {
                    for(int x = 0; x < size.x(); x++)
                        dataFile << box(Space2D(x, y)) << ",";
                }
                writePGM(rngName + "_img.pgm", detector.getHostBuffer());

                uint64_t totalNumSamples = 0;
                double mean = 0;
                uint32_t maxVal = 0;
                uint32_t minVal = static_cast<uint32_t>(-1);
                for(int y = 0; y < size.y(); y++)
                {
                    for(int x = 0; x < size.x(); x++)
                    {
                        Space2D idx(x, y);
                        uint32_t val = box(idx);
                        if(val > maxVal)
                            maxVal = val;
                        if(val < minVal)
                            minVal = val;
                        totalNumSamples += val;
                        mean += pmacc::math::linearize(size.shrink<1>(1), idx) * static_cast<uint64_t>(val);
                    }
                }
                PMACC_ASSERT(totalNumSamples == uint64_t(rngSize.productOfComponents()) * uint64_t(numSamples));
                // Expected value: (n-1)/2
                double Ex = (size.productOfComponents() - 1) / 2.;
                // Variance: (n^2 - 1) / 12
                double var = (math::pow(static_cast<double>(size.productOfComponents()), 2.0) - 1.) / 12.;
                // Mean value
                mean /= totalNumSamples;
                double errSq = 0;
                // Calc standard derivation
                for(int y = 0; y < size.y(); y++)
                {
                    for(int x = 0; x < size.x(); x++)
                    {
                        Space2D idx(x, y);
                        uint32_t val = box(idx);
                        errSq += val
                            * math::pow(
                                     static_cast<double>(pmacc::math::linearize(size.shrink<1>(1), idx) - mean),
                                     2.0);
                    }
                }
                double stdDev = sqrt(errSq / (totalNumSamples - 1));

                uint64_t avg = totalNumSamples / size.productOfComponents();
                std::cout << "  Samples: " << totalNumSamples << std::endl;
                std::cout << "      Min: " << minVal << std::endl;
                std::cout << "      Max: " << maxVal << std::endl;
                std::cout << " Avg/cell: " << avg << std::endl;
                std::cout << "     E(x): " << Ex << std::endl;
                std::cout << "     mean: " << mean << std::endl;
                std::cout << "   dev(x): " << sqrt(var) << std::endl;
                std::cout << " std. dev: " << stdDev << std::endl;
            }

        } // namespace random
    } // namespace test
} // namespace pmacc

int main(int argc, char** argv)
{
    using namespace pmacc;
    using namespace test::random;

    Environment<2>::get().initDevices(Space2D::create(1), Space2D::create(0));

    const uint32_t numSamples = (argc > 1) ? atoi(argv[1]) : 100;

    runTest<random::methods::AlpakaRand<pmacc::Acc<DIM1>>>(numSamples);

    /* finalize the pmacc context */
    Environment<>::get().finalize();
}
