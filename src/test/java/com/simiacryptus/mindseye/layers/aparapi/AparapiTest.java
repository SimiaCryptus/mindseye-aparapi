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
import com.aparapi.Kernel.EXECUTION_MODE;
import com.aparapi.Range;
import com.aparapi.device.Device;
import com.aparapi.device.OpenCLDevice;
import com.aparapi.internal.kernel.KernelManager;
import com.aparapi.internal.opencl.OpenCLPlatform;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.List;
import java.util.Random;

public class AparapiTest {
  public static final Random random = new Random();
  private static final Logger log = LoggerFactory.getLogger(AparapiTest.class);

  public AparapiTest() {
    super();
  }

  @Test
  public void main() {
    log.info("com.amd.aparapi.sample.info.Main");
    final List<OpenCLPlatform> platforms = new OpenCLPlatform().getOpenCLPlatforms();
    log.info("Machine contains " + platforms.size() + " OpenCL platforms");
    int platformc = 0;
    for (@Nonnull final OpenCLPlatform platform : platforms) {
      log.info("Platform " + platformc + "{");
      log.info("   Name    : \"" + platform.getName() + "\"");
      log.info("   Vendor  : \"" + platform.getVendor() + "\"");
      log.info("   Version : \"" + platform.getVersion() + "\"");
      final List<OpenCLDevice> devices = platform.getOpenCLDevices();
      log.info("   Platform contains " + devices.size() + " OpenCL devices");
      int devicec = 0;
      for (@Nonnull final OpenCLDevice device : devices) {
        log.info("   Device " + devicec + "{");
        log.info("       Type                  : " + device.getType());
        log.info("       GlobalMemSize         : " + device.getGlobalMemSize());
        log.info("       LocalMemSize          : " + device.getLocalMemSize());
        log.info("       MaxComputeUnits       : " + device.getMaxComputeUnits());
        log.info("       MaxWorkGroupSizes     : " + device.getMaxWorkGroupSize());
        log.info("       MaxWorkItemDimensions : " + device.getMaxWorkItemDimensions());
        log.info("   }");
        devicec++;
      }
      log.info("}");
      platformc++;
    }

    final Device bestDevice = KernelManager.instance().bestDevice();
    if (bestDevice == null) {
      log.info("OpenCLDevice.best() returned null!");
    } else {
      log.info("OpenCLDevice.best() returned { ");
      log.info("   Type                  : " + bestDevice.getType());
      log.info("   GlobalMemSize         : " + ((OpenCLDevice) bestDevice).getGlobalMemSize());
      log.info("   LocalMemSize          : " + ((OpenCLDevice) bestDevice).getLocalMemSize());
      log.info("   MaxComputeUnits       : " + ((OpenCLDevice) bestDevice).getMaxComputeUnits());
      log.info("   MaxWorkGroupSizes     : " + bestDevice.getMaxWorkGroupSize());
      log.info("   MaxWorkItemDimensions : " + bestDevice.getMaxWorkItemDimensions());
      log.info("}");
    }
  }

  @Test
  //@Ignore
  public void test1() {

    @Nonnull final OpenCLDevice openclDevice = (OpenCLDevice) Device.best();
    // final Convolution convolution = openclDevice.bind(Convolution.class);
    final AparapiTest.TestKernel testKernel = new AparapiTest.TestKernel();
    testKernel.setExecutionMode(EXECUTION_MODE.GPU);
    testKernel.setExplicit(true);
    final Range range = openclDevice.createRange3D(100, 100, 8);
    for (int j = 0; j < 2048; j++) {
      testKernel.put(testKernel.input);
      testKernel.execute(range);
      testKernel.get(testKernel.results);
      log.info("OK:" + j);
    }
    testKernel.dispose();
  }

  @Test
  public void test2() {
    @Nonnull final float inA[] = new float[1024];
    @Nonnull final float inB[] = new float[1024];
    @Nonnull final float[] result = new float[inA.length];

    @Nonnull final Kernel kernel = new Kernel() {
      @Override
      public void run() {
        final int i = getGlobalId();
        result[i] = inA[i] + inB[i];
      }
    };

    @Nonnull final Range range = Range.create(result.length);
    kernel.execute(range);
  }

  public static class TestKernel extends Kernel {

    public final int[] input = new int[10240];
    public final int[] results = new int[10240];

    @Override
    public void run() {
      final int i = getGlobalId();
      if (i > 1) {
        results[i] += (1 + results[i - 1] + results[i + 1]) * input[i];
      }
    }
  }

}
