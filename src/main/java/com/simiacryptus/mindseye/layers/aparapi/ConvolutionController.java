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

import com.simiacryptus.mindseye.lang.ComponentException;
import com.simiacryptus.ref.lang.RecycleBin;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefStringBuilder;
import com.simiacryptus.ref.wrappers.RefSystem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

public final class ConvolutionController {

  public static final int MAX_BUFFER_SIZE = 256 * 1024 * 1024;
  private static final BackpropKernel backpropTask = new BackpropKernel();
  private static final ConvolveKernel convolveTask = new ConvolveKernel();
  private static final GradientKernel kernelTask = new GradientKernel();
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ConvolutionController.class);
  private final int[] inputSize;
  @Nonnull
  private final int[] kernelSize;
  private final int[] outputSize;
  @Nullable
  private Integer paddingX = null;
  @Nullable
  private Integer paddingY = null;

  public ConvolutionController(final int[] inputSize, @Nonnull final int[] kernelSize, final Integer paddingX,
                               Integer paddingY) {
    this.inputSize = inputSize;
    this.kernelSize = kernelSize;
    this.setPaddingX(paddingX);
    this.setPaddingY(paddingY);
    outputSize = RefIntStream.range(0, kernelSize.length).map(i -> {
      int x;
      @Nullable
      Integer padding;
      if (i == 0) {
        padding = paddingX;
      } else if (i == 1) {
        padding = paddingY;
      } else {
        padding = null;
      }
      if (i == kernelSize.length - 1) {
        x = kernelSize[i] / inputSize[i];
      } else if (null == padding) {
        x = inputSize[i];
      } else {
        x = 1 + inputSize[i] - kernelSize[i] + padding;
      }
      assert 0 < x;
      return x;
    }).toArray();
    assert outputSize.length == 3;
    assert this.kernelSize.length == 3;
    assert this.inputSize.length == 3;
  }

  public int[] getOutputDims() {
    return outputSize;
  }

  @Nullable
  public Integer getPaddingX() {
    return paddingX;
  }

  public void setPaddingX(@Nullable Integer paddingX) {
    this.paddingX = paddingX;
  }

  @Nullable
  public Integer getPaddingY() {
    return paddingY;
  }

  public void setPaddingY(@Nullable Integer paddingY) {
    this.paddingY = paddingY;
  }

  public void backprop(@Nonnull final double[][] input, @Nonnull final double[] weights,
                       @Nonnull final double[][] output) {
    final int length = input.length;
    assert length == output.length;
    final int inLength = input[0].length;
    final int outLength = output[0].length;
    final int inputsPerRun = Math.min(Math.floorDiv(ConvolutionController.MAX_BUFFER_SIZE, inLength), length);
    final int runs = length / inputsPerRun;
    final int leftover = length - runs * inputsPerRun;
    try {
      synchronized (ConvolutionController.backpropTask) {
        assert 0 < weights.length;
        assert kernelSize[0] * kernelSize[1] * kernelSize[2] == weights.length;
        ConvolutionController.backpropTask.setExplicit(true);
        ConvolutionController.backpropTask.weights = weights;
        ConvolutionController.backpropTask.put(ConvolutionController.backpropTask.weights);
        ConvolutionController.backpropTask.kernelSize = kernelSize;
        ConvolutionController.backpropTask.put(ConvolutionController.backpropTask.kernelSize);
        ConvolutionController.backpropTask.kernelOffset = new int[]{
            null == paddingY ? (kernelSize[1] - 1) / 2 : paddingY,
            null == paddingX ? (kernelSize[0] - 1) / 2 : paddingX};
        ConvolutionController.backpropTask.put(ConvolutionController.convolveTask.kernelOffset);
        @Nullable
        double[] inputBuffer = null;
        @Nullable
        double[] outputBuffer = null;
        for (int run = 0; run < runs; run++) {
          final int currentIndexOffset = run * inputsPerRun;
          final int currentNumItems = leftover == 0 ? inputsPerRun : leftover;
          if (null == inputBuffer || inputBuffer.length != inLength * currentNumItems) {
            if (null != inputBuffer)
              RecycleBin.DOUBLES.recycle(inputBuffer, inputBuffer.length);
            inputBuffer = RecycleBin.DOUBLES.obtain(inLength * currentNumItems);
          }
          if (null == outputBuffer || outputBuffer.length != outLength * currentNumItems) {
            if (null != outputBuffer)
              RecycleBin.DOUBLES.recycle(outputBuffer, outputBuffer.length);
            outputBuffer = RecycleBin.DOUBLES.obtain(outLength * currentNumItems);
          }
          for (int i = 0; i < currentNumItems; i++) {
            assert outLength == output[currentIndexOffset + i].length;
            RefSystem.arraycopy(output[currentIndexOffset + i], 0, outputBuffer,
                i * outLength, outLength);
          }
          assert 0 < inputBuffer.length;
          assert 0 < outputBuffer.length;
          ConvolutionController.backpropTask.input = inputBuffer;
          ConvolutionController.backpropTask.output = outputBuffer;
          ConvolutionController.backpropTask.outputSize = outputSize;
          ConvolutionController.backpropTask.inputSize = inputSize;
          ConvolutionController.backpropTask.put(ConvolutionController.backpropTask.outputSize);
          ConvolutionController.backpropTask.put(ConvolutionController.backpropTask.inputSize);
          ConvolutionController.backpropTask.put(ConvolutionController.backpropTask.output);
          ConvolutionController.backpropTask.exe();
          ConvolutionController.backpropTask.get(ConvolutionController.backpropTask.input);
          ConvolutionController.backpropTask.input = null;
          ConvolutionController.backpropTask.output = null;
          ConvolutionController.backpropTask.outputSize = null;
          ConvolutionController.backpropTask.inputSize = null;
          for (int i = 0; i < currentNumItems; i++) {
            assert inLength == input[currentIndexOffset + i].length;
            RefSystem.arraycopy(inputBuffer, i * inLength,
                input[currentIndexOffset + i], 0, inLength);
          }
        }
        assert inputBuffer != null;
        RecycleBin.DOUBLES.recycle(inputBuffer, inputBuffer.length);
        RecycleBin.DOUBLES.recycle(outputBuffer, outputBuffer.length);
        ConvolutionController.backpropTask.kernelSize = null;
        ConvolutionController.backpropTask.weights = null;
      }
    } catch (@Nonnull final Throwable e) {
      throw new ComponentException("Error apply " + this, e);
    }
  }

  public void convolve(@Nonnull final double[][] input, @Nonnull final double[] weights,
                       @Nonnull final double[][] output) {
    final int length = input.length;
    assert length == output.length;
    final int inLength = input[0].length;
    final int outLength = output[0].length;
    final int inputsPerRun = Math.min(Math.floorDiv(ConvolutionController.MAX_BUFFER_SIZE, inLength), length);
    assert 0 < inputsPerRun : "Requested buffer is over max of " + ConvolutionController.MAX_BUFFER_SIZE;
    final int runs = length / inputsPerRun;
    final int leftover = length - runs * inputsPerRun;
    try {
      synchronized (ConvolutionController.convolveTask) {
        assert 0 < weights.length;
        ConvolutionController.convolveTask.setExplicit(true);
        ConvolutionController.convolveTask.weights = weights;
        ConvolutionController.convolveTask.put(ConvolutionController.convolveTask.weights);
        ConvolutionController.convolveTask.kernelSize = kernelSize;
        ConvolutionController.convolveTask.kernelOffset = new int[]{
            null == paddingY ? (kernelSize[1] - 1) / 2 : paddingY,
            null == paddingX ? (kernelSize[0] - 1) / 2 : paddingX};
        ConvolutionController.convolveTask.put(ConvolutionController.convolveTask.kernelOffset);
        ConvolutionController.convolveTask.put(ConvolutionController.convolveTask.kernelSize);
        @Nullable
        double[] inputBuffer = null;
        @Nullable
        double[] outputBuffer = null;
        for (int run = 0; run <= runs; run++) {
          final int currentIndexOffset = run * inputsPerRun;
          final int currentNumItems = run < runs ? inputsPerRun : leftover;
          if (0 == currentNumItems) {
            continue;
          }
          if (null == inputBuffer || inputBuffer.length != inLength * currentNumItems) {
            if (null != inputBuffer)
              RecycleBin.DOUBLES.recycle(inputBuffer, inputBuffer.length);
            inputBuffer = RecycleBin.DOUBLES.obtain(inLength * currentNumItems);
          }
          if (null == outputBuffer || outputBuffer.length != outLength * currentNumItems) {
            if (null != outputBuffer)
              RecycleBin.DOUBLES.recycle(outputBuffer, outputBuffer.length);
            outputBuffer = RecycleBin.DOUBLES.obtain(outLength * currentNumItems);
          }
          for (int i = 0; i < currentNumItems; i++) {
            assert inLength == input[currentIndexOffset + i].length;
            RefSystem.arraycopy(input[currentIndexOffset + i], 0, inputBuffer,
                i * inLength, inLength);
          }
          assert 0 < inputBuffer.length;
          assert 0 < outputBuffer.length;
          ConvolutionController.convolveTask.input = inputBuffer;
          ConvolutionController.convolveTask.output = outputBuffer;
          ConvolutionController.convolveTask.outputSize = outputSize;
          ConvolutionController.convolveTask.inputSize = inputSize;
          ConvolutionController.convolveTask.put(ConvolutionController.convolveTask.outputSize);
          ConvolutionController.convolveTask.put(ConvolutionController.convolveTask.inputSize);
          ConvolutionController.convolveTask.put(ConvolutionController.convolveTask.input);
          ConvolutionController.convolveTask.exe();
          ConvolutionController.convolveTask.get(ConvolutionController.convolveTask.output);
          ConvolutionController.convolveTask.input = null;
          ConvolutionController.convolveTask.output = null;
          ConvolutionController.convolveTask.outputSize = null;
          ConvolutionController.convolveTask.inputSize = null;
          for (int i = 0; i < currentNumItems; i++) {
            assert outLength == output[currentIndexOffset + i].length;
            RefSystem.arraycopy(outputBuffer, i * outLength,
                output[currentIndexOffset + i], 0, outLength);
          }
        }
        assert inputBuffer != null;
        RecycleBin.DOUBLES.recycle(inputBuffer, inputBuffer.length);
        RecycleBin.DOUBLES.recycle(outputBuffer, outputBuffer.length);
        ConvolutionController.convolveTask.kernelSize = null;
        ConvolutionController.convolveTask.weights = null;
      }
    } catch (@Nonnull final Throwable e) {
      throw new ComponentException("Error apply " + this, e);
    }
  }

  public void gradient(@Nonnull final double[][] input, @Nonnull final double[] weights,
                       @Nonnull final double[][] output) {
    final int length = input.length;
    assert length == output.length;
    final int inLength = input[0].length;
    final int outLength = output[0].length;
    final int inputsPerRun = Math
        .min(Math.floorDiv(ConvolutionController.MAX_BUFFER_SIZE, Math.max(inLength, outLength)), length);
    final int runs = length / inputsPerRun;
    final int leftover = length - runs * inputsPerRun;
    @Nullable
    double[] inputBuffer = null;
    @Nullable
    double[] outputBuffer = null;
    for (int run = 0; run < runs; run++) {
      final int currentIndexOffset = run * inputsPerRun;
      final int currentNumItems = leftover == 0 ? inputsPerRun : leftover;
      if (null == inputBuffer || inputBuffer.length != inLength * currentNumItems) {
        if (null != inputBuffer)
          RecycleBin.DOUBLES.recycle(inputBuffer, inputBuffer.length);
        inputBuffer = RecycleBin.DOUBLES.obtain(inLength * currentNumItems);
      }
      if (null == outputBuffer || outputBuffer.length != outLength * currentNumItems) {
        if (null != outputBuffer)
          RecycleBin.DOUBLES.recycle(outputBuffer, outputBuffer.length);
        outputBuffer = RecycleBin.DOUBLES.obtain(outLength * currentNumItems);
      }
      for (int i = 0; i < currentNumItems; i++) {
        assert inLength == input[currentIndexOffset + i].length;
        assert outLength == output[currentIndexOffset + i].length;
        RefSystem.arraycopy(input[currentIndexOffset + i], 0, inputBuffer, i * inLength,
            inLength);
        RefSystem.arraycopy(output[currentIndexOffset + i], 0, outputBuffer,
            i * outLength, outLength);
      }
      final int parallelism = Math.min(16, inLength);
      final double[] buffer = RecycleBin.DOUBLES.obtain(weights.length * parallelism);
      gradient(inputBuffer, buffer, weights.length, outputBuffer);
      RefIntStream.range(0, weights.length).forEach(weightIndex -> {
        for (int i = weightIndex; i < buffer.length; i += weights.length) {
          weights[weightIndex] += buffer[i];
        }
      });
      RecycleBin.DOUBLES.recycle(buffer, buffer.length);
    }
    assert inputBuffer != null;
    RecycleBin.DOUBLES.recycle(inputBuffer, inputBuffer.length);
    RecycleBin.DOUBLES.recycle(outputBuffer, outputBuffer.length);
  }

  @Override
  public String toString() {
    @Nonnull final RefStringBuilder builder = new RefStringBuilder();
    builder.append("Convolve [");
    builder.append(RefArrays.toString(inputSize));
    builder.append(" x ");
    builder.append(RefArrays.toString(kernelSize));
    builder.append(" => ");
    builder.append(RefArrays.toString(outputSize));
    builder.append("]");
    return builder.toString();
  }

  private void gradient(@Nonnull final double[] input, @Nonnull final double[] weights, final int weightSize,
                        @Nonnull final double[] output) {
    assert 0 < input.length;
    assert 0 < weights.length;
    assert 0 < output.length;
    try {
      synchronized (ConvolutionController.kernelTask) {
        ConvolutionController.kernelTask.input = input;
        ConvolutionController.kernelTask.weights = weights;
        ConvolutionController.kernelTask.output = output;
        ConvolutionController.kernelTask.outputSize = outputSize;
        ConvolutionController.kernelTask.inputSize = inputSize;
        ConvolutionController.kernelTask.kernelSize = kernelSize;
        ConvolutionController.kernelTask.weightSize = weightSize;
        ConvolutionController.kernelTask.paralellism = weights.length / weightSize;
        ConvolutionController.kernelTask.kernelOffset = new int[]{
            paddingY == null ? (kernelSize[1] - 1) / 2 : paddingY,
            paddingX == null ? (kernelSize[0] - 1) / 2 : paddingX};
        ConvolutionController.kernelTask.setExplicit(true);
        ConvolutionController.kernelTask.put(ConvolutionController.convolveTask.kernelOffset);
        ConvolutionController.kernelTask.put(ConvolutionController.kernelTask.outputSize);
        ConvolutionController.kernelTask.put(ConvolutionController.kernelTask.inputSize);
        ConvolutionController.kernelTask.put(ConvolutionController.kernelTask.kernelSize);
        ConvolutionController.kernelTask.put(ConvolutionController.kernelTask.input);
        ConvolutionController.kernelTask.put(ConvolutionController.kernelTask.output);
        ConvolutionController.kernelTask.exe();
        ConvolutionController.kernelTask.get(ConvolutionController.kernelTask.weights);
        ConvolutionController.kernelTask.input = null;
        ConvolutionController.kernelTask.weights = null;
        ConvolutionController.kernelTask.output = null;
        ConvolutionController.kernelTask.outputSize = null;
        ConvolutionController.kernelTask.inputSize = null;
        ConvolutionController.kernelTask.kernelSize = null;
      }
    } catch (@Nonnull final Throwable e) {
      throw new ComponentException("Error apply " + this, e);
    }
  }
}
