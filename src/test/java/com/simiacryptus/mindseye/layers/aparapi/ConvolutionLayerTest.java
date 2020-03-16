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

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.test.LayerTestBase;

import javax.annotation.Nonnull;

public abstract class ConvolutionLayerTest extends LayerTestBase {

  public static class Basic extends ConvolutionLayerTest {

    private final int inputBands = 1;
    private final int outputBands = 1;

    @Nonnull
    @Override
    public int[][] getLargeDims() {

      return new int[][]{{200, 200, inputBands}};
    }

    @Nonnull
    @Override
    public Layer getLayer() {
      ConvolutionLayer temp_01_0002 = new ConvolutionLayer(3, 3, inputBands, outputBands, true);
      temp_01_0002.setWeights(() -> this.random());
      ConvolutionLayer temp_01_0001 = temp_01_0002.addRef();
      temp_01_0002.freeRef();
      return temp_01_0001;
    }

    @Nonnull
    @Override
    public int[][] getSmallDims() {
      return new int[][]{{8, 8, 1}};
    }

  }

  public static class Downsize extends ConvolutionLayerTest {

    @Nonnull
    @Override
    public Layer getLayer() {
      ConvolutionLayer temp_01_0004 = new ConvolutionLayer(3, 3, 7, 3, false);
      temp_01_0004.setWeights(() -> this.random());
      ConvolutionLayer temp_01_0003 = temp_01_0004.addRef();
      temp_01_0004.freeRef();
      return temp_01_0003;
    }

    @Nonnull
    @Override
    public int[][] getSmallDims() {
      return new int[][]{{3, 3, 7}};
    }

  }

  public static class Upsize extends ConvolutionLayerTest {

    @Nonnull
    @Override
    public Layer getLayer() {
      ConvolutionLayer temp_01_0006 = new ConvolutionLayer(3, 3, 2, 3, false);
      temp_01_0006.setWeights(() -> this.random());
      ConvolutionLayer temp_01_0005 = temp_01_0006.addRef();
      temp_01_0006.freeRef();
      return temp_01_0005;
    }

    @Nonnull
    @Override
    public int[][] getSmallDims() {
      return new int[][]{{3, 3, 2}};
    }

  }
}
