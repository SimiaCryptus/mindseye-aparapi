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

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.DoubleSupplier;
import java.util.function.ToDoubleFunction;

@SuppressWarnings("serial")
public @RefAware
class ConvolutionLayer extends LayerBase {

  @Nullable
  public final Tensor kernel;
  @Nullable
  private Integer paddingX = null;
  @Nullable
  private Integer paddingY = null;

  protected ConvolutionLayer() {
    this(null, true);
  }

  public ConvolutionLayer(final int width, final int height, final int bands) {
    this(width, height, bands, true);
  }

  public ConvolutionLayer(final int width, final int height, final int bands, final boolean simple) {
    this(new Tensor(width, height, bands), simple);
    assert !simple || 0 == (width - 1) % 2 : "Simple kernels must have odd width";
    assert !simple || 0 == (height - 1) % 2 : "Simple kernels must have odd height";
  }

  public ConvolutionLayer(final int width, final int height, final int inputBands, final int outputBands) {
    this(width, height, inputBands * outputBands);
  }

  public ConvolutionLayer(final int width, final int height, final int inputBands, final int outputBands,
                          final boolean simple) {
    this(width, height, inputBands * outputBands, simple);
  }

  protected ConvolutionLayer(@Nonnull final JsonObject json,
                             Map<CharSequence, byte[]> resources) {
    super(json);
    kernel = Tensor.fromJson(json.get("filter"), resources);
    JsonElement paddingX = json.get("paddingX");
    if (null != paddingX && paddingX.isJsonPrimitive())
      this.setPaddingX((paddingX.getAsInt()));
    JsonElement paddingY = json.get("paddingY");
    if (null != paddingY && paddingY.isJsonPrimitive())
      this.setPaddingY((paddingY.getAsInt()));
  }

  protected ConvolutionLayer(@Nonnull final Tensor kernel, final boolean simple) {
    super();
    this.paddingX = simple ? null : 0;
    this.paddingY = simple ? null : 0;
    @Nonnull
    int[] dimensions = kernel.getDimensions();
    if (dimensions.length != 3)
      throw new IllegalArgumentException(RefArrays.toString(dimensions));
    if (dimensions[0] <= 0)
      throw new IllegalArgumentException(RefArrays.toString(dimensions));
    if (dimensions[1] <= 0)
      throw new IllegalArgumentException(RefArrays.toString(dimensions));
    if (dimensions[2] <= 0)
      throw new IllegalArgumentException(RefArrays.toString(dimensions));
    if (dimensions[2] <= 0)
      throw new IllegalArgumentException(RefArrays.toString(dimensions));
    this.kernel = kernel;
  }

  @Nullable
  public Integer getPaddingX() {
    return paddingX;
  }

  @Nonnull
  public void setPaddingX(Integer paddingX) {
    this.paddingX = paddingX;
  }

  @Nullable
  public Integer getPaddingY() {
    return paddingY;
  }

  @Nonnull
  public void setPaddingY(Integer paddingY) {
    this.paddingY = paddingY;
  }

  @Nonnull
  public ConvolutionLayer setWeights(@Nonnull final DoubleSupplier f) {
    kernel.coordStream(true).forEach(c -> {
      kernel.set(c, f.getAsDouble());
    });
    return this;
  }

  @Nonnull
  public ConvolutionLayer setWeights(@Nonnull final ToDoubleFunction<Coordinate> f) {
    kernel.coordStream(true).forEach(c -> {
      kernel.set(c, f.applyAsDouble(c));
    });
    return this;
  }

  @SuppressWarnings("unused")
  public static ConvolutionLayer fromJson(@Nonnull final JsonObject json,
                                          Map<CharSequence, byte[]> rs) {
    return new ConvolutionLayer(json, rs);
  }

  public static @SuppressWarnings("unused")
  ConvolutionLayer[] addRefs(ConvolutionLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ConvolutionLayer::addRef)
        .toArray((x) -> new ConvolutionLayer[x]);
  }

  public static @SuppressWarnings("unused")
  ConvolutionLayer[][] addRefs(ConvolutionLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ConvolutionLayer::addRefs)
        .toArray((x) -> new ConvolutionLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final Result input = inObj[0];
    final TensorList batch = input.getData();
    @Nonnull final int[] inputDims = batch.get(0).getDimensions();
    @Nonnull final int[] kernelDims = kernel.getDimensions();
    final ConvolutionLayer convolutionLayer = ConvolutionLayer.this;
    @Nullable final double[] kernelData = convolutionLayer.kernel.getData();
    @Nonnull final ConvolutionController convolutionController = new ConvolutionController(inputDims, kernelDims, paddingX,
        paddingY);
    final Tensor[] output = RefIntStream.range(0, batch.length())
        .mapToObj(dataIndex -> new Tensor(convolutionController.getOutputDims())).toArray(i -> new Tensor[i]);
    try {
      final double[][] inputBuffers = batch.stream().map(x -> {
        @Nullable
        double[] data = x.getData();
        x.detach();
        return data;
      }).toArray(i -> new double[i][]);
      final double[][] outputBuffers = RefArrays.stream(output).map(x -> x.getData())
          .toArray(i -> new double[i][]);
      convolutionController.convolve(inputBuffers, kernelData, outputBuffers);
    } catch (@Nonnull final Throwable e) {
      throw new RuntimeException(
          "Error mapCoords png res " + RefArrays.toString(inputDims), e);
    }
    int outputLength = output.length;
    return new Result(new TensorArray(output),
        new Result.Accumulator() {
          @Override
          public void accept(DeltaSet<UUID> buffer, TensorList error) {
            if (!ConvolutionLayer.this.isFrozen()) {
              final double[][] inputBuffers = batch.stream().map(x -> {
                return x.getData();
              }).toArray(i -> new double[i][]);
              final double[][] outputBuffers = error.stream().map(x -> {
                return x.getData();
              }).toArray(i -> new double[i][]);
              @Nonnull final Tensor weightGradient = new Tensor(kernelDims);
              convolutionController.gradient(inputBuffers, weightGradient.getData(), outputBuffers);

              buffer.get(convolutionLayer.getId(), kernelData).addInPlace(weightGradient.getData());
            }
            if (input.isAlive()) {
              final Tensor[] inputBufferTensors = RefIntStream.range(0, outputLength)
                  .mapToObj(dataIndex -> new Tensor(inputDims)).toArray(i -> new Tensor[i]);
              final double[][] inputBuffers = RefArrays.stream(inputBufferTensors)
                  .map(x -> {
                    return x.getData();
                  }).toArray(i -> new double[i][]);
              final double[][] outputBuffers = error.stream().map(x -> {
                return x.getData();
              }).toArray(i -> new double[i][]);
              convolutionController.backprop(inputBuffers, kernelData, outputBuffers);
              @Nonnull
              TensorArray tensorArray = new TensorArray(inputBufferTensors);
              input.accumulate(buffer, tensorArray);
            }
          }
        }) {

      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }

      public void _free() {
      }
    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources,
                            @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.add("filter", kernel.getJson(resources, dataSerializer));
    JsonElement paddingX = json.get("paddingX");
    if (null != paddingX && paddingX.isJsonPrimitive())
      this.setPaddingX((paddingX.getAsInt()));
    JsonElement paddingY = json.get("paddingY");
    if (null != paddingY && paddingY.isJsonPrimitive())
      this.setPaddingY((paddingY.getAsInt()));
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList(kernel.getData());
  }

  public void _free() {
    super._free();
  }

  public @Override
  @SuppressWarnings("unused")
  ConvolutionLayer addRef() {
    return (ConvolutionLayer) super.addRef();
  }
}
