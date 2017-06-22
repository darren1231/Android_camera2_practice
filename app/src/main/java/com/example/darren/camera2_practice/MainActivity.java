package com.example.darren.camera2_practice;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.ImageReader;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.annotation.NonNull;
import android.support.annotation.RequiresApi;
import android.support.annotation.StringDef;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.util.SparseIntArray;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import static android.R.attr.width;

@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
public class MainActivity extends AppCompatActivity {

    // Tensorflow
    final String MODEL_FILE = "file:///android_asset/test_optimized_detect_smoke.pb";
    private static final int CROP_SIZE = 64;
    private static final String INPUT_NODE = "smoke_input";
    private static final String KEEP_NODE = "keep_prob";
    private static final int[] KEEP_SIZE = {1};
    private static final float[] KEEP_VALUE = {1.0f};
    private static final String OUTPUT_NODE = "smoke_output";
    private static final int[] INPUT_SIZE = {1,CROP_SIZE,CROP_SIZE,3};
    private int RGBintvalues [] = new int[CROP_SIZE * CROP_SIZE];
    private float RGBfloatValues[] = new float[CROP_SIZE*CROP_SIZE*3];
    float[] CNN_output = {0, 0};
    private Bitmap mPreviewBitmap;
    private Bitmap CropBitmap;
    private TensorFlowInferenceInterface inferenceInterface;
    static {
        System.loadLibrary("tensorflow_inference");
    }
    long time_start=0, time_end=0,time_process=0;


    private static final int REQUEST_CAMERA_PERMISSION_RESULT=0;
    private TextureView mTextureView;
    private TextView mTextView;
    private TextureView.SurfaceTextureListener mSurfaceTextureListener = new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
            //Toast.makeText(getApplicationContext(),"TextureView is available", Toast.LENGTH_SHORT).show();
            setupCamera(width,height);
            connectCamera();
        }

        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {

        }

        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
            return false;
        }

        boolean bProcessDone = true;
        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surface) {
            mTextView.setText("Probability--\n"+CNN_output[0]+"\n"+CNN_output[1]);
            if(bProcessDone) {
                bProcessDone = false;
                new Thread(new Runnable() {
                    @Override
                    public void run() {

                        // Scaling
                        mPreviewBitmap=mTextureView.getBitmap();
                        //mPreviewBitmap = Bitmap.createScaledBitmap(mPreviewBitmap, SIZE_width, SIZE_hight, false);
                        //imv.setImageBitmap(mPreviewBitmap);
                        CropBitmap = Bitmap.createBitmap(mPreviewBitmap,0, 0, CROP_SIZE,CROP_SIZE);
                        CropBitmap.getPixels(RGBintvalues, 0, CropBitmap.getWidth(), 0, 0, CropBitmap.getWidth(), CropBitmap.getHeight());
                        time_start=System.currentTimeMillis();
                        int val = 0;
                        for (int i = 0; i < RGBintvalues.length; ++i) {
                            val = RGBintvalues[i];

                            RGBfloatValues[i * 3 + 0] = (float) ((val >> 16) & 0xFF)/255;
                            RGBfloatValues[i * 3 + 1] = (float) ((val >> 8) & 0xFF)/255;
                            RGBfloatValues[i * 3 + 2] = (float) (val & 0xFF)/255;
                            //mTextView.setText(Integer.toString(test));
                        }
                        inferenceInterface.fillNodeFloat(INPUT_NODE, INPUT_SIZE, RGBfloatValues);
                        inferenceInterface.fillNodeFloat(KEEP_NODE, KEEP_SIZE, KEEP_VALUE);
                        inferenceInterface.runInference(new String[]{OUTPUT_NODE});

                        inferenceInterface.readNodeFloat(OUTPUT_NODE, CNN_output);

                        time_end=System.currentTimeMillis();
                        time_process = time_end-time_start;
                        Log.d("RGB","Process Time:"+time_process+"");
//                        mTextView.setText("Probability--"+CNN_output[0]+","+CNN_output[1]);
                        // Tensorflow
                        // return to UI

                        bProcessDone = true;
//                        test++;
//                        mTextView.setText(Integer.toString(test));
                    }
                }).start();
            } else {
                //DO nothing
            }

        }// surface updated
    };

    //  Teach you how to get the real camera device into Android studio
    private CameraDevice mCameraDevice;
    private CameraDevice.StateCallback mCameraDeviceStateCallback = new CameraDevice.StateCallback(){

        @Override
        public void onOpened(@NonNull CameraDevice camera) {
            mCameraDevice = camera;
            //Toast.makeText(getApplicationContext(),"Camera connection open!!",Toast.LENGTH_SHORT).show();
            startPreview();
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice camera) {
            camera.close();
            mCameraDevice=null;

        }

        @Override
        public void onError(@NonNull CameraDevice camera, int error) {
            camera.close();
            mCameraDevice=null;
        }
    };
    private HandlerThread mBackgroundHandlerThread;
    private Handler mBackgroundHandler;
    private String mCameraID;
    private Size mPreviewSize;
    private CaptureRequest.Builder mCaptureRequestBuilder;

    private static SparseIntArray ORIENTATIONS = new SparseIntArray();
    static {
        ORIENTATIONS.append(Surface.ROTATION_0, 0);
        ORIENTATIONS.append(Surface.ROTATION_90, 90);
        ORIENTATIONS.append(Surface.ROTATION_180, 180);
        ORIENTATIONS.append(Surface.ROTATION_270, 270);
    }

    private static class CompareSizeByArea implements Comparator<Size> {

        @Override
        public int compare(Size lhs, Size rhs) {
            return Long.signum( (long)(lhs.getWidth() * lhs.getHeight()) -
                    (long)(rhs.getWidth() * rhs.getHeight()));
        }
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mTextureView =(TextureView)findViewById(R.id.textureView);
        mTextView = (TextView)findViewById(R.id.textView);

        inferenceInterface = new TensorFlowInferenceInterface();
        inferenceInterface.initializeTensorFlow(getAssets(), MODEL_FILE);
    }

    // if not available, set it to the listener, then would go to "onSurfaceTextureAvailable"
    @Override
    protected void onResume() {
        super.onResume();
        startBackgroundThread();
        if (mTextureView.isAvailable()) {
            setupCamera(mTextureView.getWidth(),mTextureView.getHeight());
            connectCamera();
        } else {
            mTextureView.setSurfaceTextureListener(mSurfaceTextureListener);
        }
    }

    //  This is done after user pick the permission selection
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if(requestCode == REQUEST_CAMERA_PERMISSION_RESULT) {
            if(grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(getApplicationContext(),
                        "Application will not run without camera services", Toast.LENGTH_SHORT).show();
            }
        }
    }

    protected  void onPause() {
        closeCamera();
        stopBackgroundThread();
        super.onPause();
    }

    // onWindowFocusChanged----- full screen
    @Override
    public void onWindowFocusChanged(boolean hasFocas) {
        super.onWindowFocusChanged(hasFocas);
        View decorView = getWindow().getDecorView();
        if (hasFocas) {
            decorView.setSystemUiVisibility(View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                    | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                    | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                    | View.SYSTEM_UI_FLAG_FULLSCREEN
                    | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION);
        }
    }

    // We need to use get  CameraManager to setupCamera, and we will skip the front camera, so we will get the back camera
    // Notice: this function need to be in try function, this function need to put in onResume method and onSurfaceTextureAvailable
    // method
    private void setupCamera(int width, int height) {
        CameraManager cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            for(String cameraId : cameraManager.getCameraIdList()){
                // Put ID in the CameraCharacteristics and we can get it
                CameraCharacteristics cameraCharacteristics = cameraManager.getCameraCharacteristics(cameraId);
                if(cameraCharacteristics.get(CameraCharacteristics.LENS_FACING) ==
                        CameraCharacteristics.LENS_FACING_FRONT){
                        continue;
                }

                // map contain many information, such as output size
                StreamConfigurationMap map = cameraCharacteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
                int deviceOrientation = getWindowManager().getDefaultDisplay().getRotation();
                int totalRotation = sensorToDeviceRotation(cameraCharacteristics, deviceOrientation);
                boolean swapRotation = totalRotation == 90 || totalRotation == 270;
                int rotatedWidth = width;
                int rotatedHeight = height;

                // Detect whether Rotation
                if(swapRotation) {
                    rotatedWidth = height;
                    rotatedHeight = width;
                }
                mPreviewSize = chooseOptimalSize(map.getOutputSizes(SurfaceTexture.class), rotatedWidth, rotatedHeight);
                mCameraID = cameraId;
                return;
            }
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    // use cameraID and cameraManager.openCamera to openCamera , callback used to opencamera
    // Notice:  we have to check the permission
    private void connectCamera() {
        CameraManager cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                if(ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) ==
                        PackageManager.PERMISSION_GRANTED) {
                    cameraManager.openCamera(mCameraID, mCameraDeviceStateCallback, mBackgroundHandler);
                } else {
                    if(shouldShowRequestPermissionRationale(android.Manifest.permission.CAMERA)) {
                        Toast.makeText(this,"Video app required access to camera", Toast.LENGTH_SHORT).show();
                    }
                    requestPermissions(new String[] {android.Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO
                    }, REQUEST_CAMERA_PERMISSION_RESULT);
                }

            } else {
                cameraManager.openCamera(mCameraID, mCameraDeviceStateCallback, mBackgroundHandler);
            }
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }


    // Used in the Onopen method
    private void startPreview() {
        SurfaceTexture surfaceTexture = mTextureView.getSurfaceTexture();
        surfaceTexture.setDefaultBufferSize(mPreviewSize.getWidth(), mPreviewSize.getHeight());
        Surface previewSurface = new Surface(surfaceTexture);

        try {
            mCaptureRequestBuilder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            mCaptureRequestBuilder.addTarget(previewSurface);

            mCameraDevice.createCaptureSession(Arrays.asList(previewSurface),
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(CameraCaptureSession session) {
                            //Log.d(TAG, "onConfigured: startPreview");
                            try {
                                session.setRepeatingRequest(mCaptureRequestBuilder.build(),
                                        null, mBackgroundHandler);
                            } catch (CameraAccessException e) {
                                e.printStackTrace();
                            }
                        }

                        @Override
                        public void onConfigureFailed(CameraCaptureSession session) {
                            Toast.makeText(getApplicationContext(),"Unable to setup camera preview",Toast.LENGTH_SHORT).show();
                            //Log.d(TAG, "onConfigureFailed: startPreview");

                        }
                    }, null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }



    // add my own closeCamera method, used in the onPause method
    private void closeCamera() {
        if (mCameraDevice != null) {
            mCameraDevice.close();
            mCameraDevice = null;
        }
    }

    // Put it in the onResume method
    private void startBackgroundThread() {
        mBackgroundHandlerThread = new HandlerThread("Camera2VideoImage");
        mBackgroundHandlerThread.start();
        mBackgroundHandler = new Handler(mBackgroundHandlerThread.getLooper());
    }


    // Put it in the onPause method
    private void stopBackgroundThread() {
        mBackgroundHandlerThread.quitSafely();
        try {
            mBackgroundHandlerThread.join();
            mBackgroundHandlerThread = null;
            mBackgroundHandler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    // To keep camera rotation,  input camera id and rotation parameter. This function used in setupcamera method and can
    // caculate total rotation
    private static int sensorToDeviceRotation(CameraCharacteristics cameraCharacteristics, int deviceOrientation) {
        int sensorOrienatation = cameraCharacteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
        deviceOrientation = ORIENTATIONS.get(deviceOrientation);
        return (sensorOrienatation + deviceOrientation + 360) % 360;
    }

    // Used in the setup camera
    private static Size chooseOptimalSize(Size[] choices, int width, int height) {
        List<Size> bigEnough = new ArrayList<Size>();
        for(Size option : choices) {
            if(option.getHeight() == option.getWidth() * height / width &&
                    option.getWidth() >= width && option.getHeight() >= height) {
                bigEnough.add(option);
            }
        }
        if(bigEnough.size() > 0) {
            return Collections.min(bigEnough, new CompareSizeByArea());
        } else {
            return choices[0];
        }
    }
}
