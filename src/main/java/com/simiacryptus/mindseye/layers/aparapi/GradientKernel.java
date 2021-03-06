/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.aparapi;

import com.aparapi.Kernel;
import com.aparapi.Range;

import javax.annotation.Nullable;

public final class GradientKernel extends Kernel {

  @Nullable
  public double[] input;
  @Nullable
  public int[] inputSize;
  public int[] kernelOffset;
  @Nullable
  public int[] kernelSize;
  @Nullable
  public double[] output;
  @Nullable
  public int[] outputSize;
  public int paralellism;
  @Nullable
  public double[] weights;
  public int weightSize;

  public GradientKernel() {
  }

  public void exe() {
    if (null == kernelSize)
      throw new IllegalStateException();
    execute(Range.create(weightSize, paralellism));
  }

  @Override
  public void run() {
    final int k = getGlobalId(0);
    final int threadNumber = getGlobalId(1);
    assert kernelSize != null;
    final int ks0 = kernelSize[0];
    final int ks1 = ks0 * kernelSize[1];
    final int k2 = k / ks1;
    final int k1 = k % ks1 / ks0;
    final int k0 = k % ks0;

    double accum = 0.;
    assert input != null;
    for (int i = threadNumber; i < input.length; i += paralellism) {
      if (0. != input[i]) {
        assert inputSize != null;
        final int is0 = inputSize[0];
        final int is1 = is0 * inputSize[1];
        final int is2 = is1 * inputSize[2];
        final int batch = i / is2;
        final int i2 = i % is2 / is1;
        final int i1 = i % is1 / is0;
        final int i0 = i % is0;

        assert outputSize != null;
        final int o2 = k2 - i2 * outputSize[2];
        if (o2 >= 0 && o2 < outputSize[2]) {
          final int o1 = i1 + k1 - kernelOffset[1];
          final int o0 = i0 + k0 - kernelOffset[0];
          if (o0 < outputSize[0] && o1 < outputSize[1] && o0 >= 0 && o1 >= 0) {
            final int o = o0 + outputSize[0] * (o1 + outputSize[1] * (o2 + outputSize[2] * batch));
            assert output != null;
            accum += input[i] * output[o];
          }
        }
      }
    }
    assert weights != null;
    weights[k + weightSize * threadNumber] = accum;
  }

}