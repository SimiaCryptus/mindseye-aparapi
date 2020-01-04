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
import com.simiacryptus.mindseye.layers.java.LayerTestBase;

import javax.annotation.Nonnull;
import java.util.Random;

public abstract @com.simiacryptus.ref.lang.RefAware class ConvolutionLayerTest extends LayerTestBase {

  public static @com.simiacryptus.ref.lang.RefAware class Basic extends ConvolutionLayerTest {

    private final int inputBands = 1;
    private final int outputBands = 1;

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new ConvolutionLayer(3, 3, inputBands, outputBands, true).setWeights(() -> this.random());
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][] { { 8, 8, 1 } };
    }

    @Nonnull
    @Override
    public int[][] getLargeDims(Random random) {

      return new int[][] { { 200, 200, inputBands } };
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Basic addRef() {
      return (Basic) super.addRef();
    }

    public static @SuppressWarnings("unused") Basic[] addRefs(Basic[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Basic::addRef).toArray((x) -> new Basic[x]);
    }

  }

  public static @com.simiacryptus.ref.lang.RefAware class Downsize extends ConvolutionLayerTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][] { { 3, 3, 7 } };
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new ConvolutionLayer(3, 3, 7, 3, false).setWeights(() -> this.random());
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Downsize addRef() {
      return (Downsize) super.addRef();
    }

    public static @SuppressWarnings("unused") Downsize[] addRefs(Downsize[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Downsize::addRef)
          .toArray((x) -> new Downsize[x]);
    }

  }

  public static @com.simiacryptus.ref.lang.RefAware class Upsize extends ConvolutionLayerTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][] { { 3, 3, 2 } };
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new ConvolutionLayer(3, 3, 2, 3, false).setWeights(() -> this.random());
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Upsize addRef() {
      return (Upsize) super.addRef();
    }

    public static @SuppressWarnings("unused") Upsize[] addRefs(Upsize[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Upsize::addRef).toArray((x) -> new Upsize[x]);
    }

  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") ConvolutionLayerTest addRef() {
    return (ConvolutionLayerTest) super.addRef();
  }

  public static @SuppressWarnings("unused") ConvolutionLayerTest[] addRefs(ConvolutionLayerTest[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ConvolutionLayerTest::addRef)
        .toArray((x) -> new ConvolutionLayerTest[x]);
  }

  public static @SuppressWarnings("unused") ConvolutionLayerTest[][] addRefs(ConvolutionLayerTest[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ConvolutionLayerTest::addRefs)
        .toArray((x) -> new ConvolutionLayerTest[x][]);
  }
}
