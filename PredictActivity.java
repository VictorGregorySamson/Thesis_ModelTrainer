package com.example.goodsign;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageButton;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;

import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;
import androidx.lifecycle.ViewModelProvider;
import com.google.common.util.concurrent.ListenableFuture;

import com.google.mediapipe.tasks.vision.core.RunningMode;

import android.os.Handler;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.widget.PopupMenu;
import android.widget.Toast;


import org.json.JSONArray;
import org.json.JSONException;

import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class PredictActivity extends AppCompatActivity implements HolisticLandmarkerHelper.LandmarkerListener, ModelPredictor.PredictionListener {

    private static final String TAG = "PredictActivity";
    private static final int CAMERA_PERMISSION_REQUEST_CODE = 100;
    private static final int FRAME_BUFFER_SIZE = 120;

    private ModelPredictor modelPredictor;
    private TextView predictionText;
    private ImageButton audioButton;
    private TextToSpeech textToSpeech;
    private int audioMode = 0; // 0: off, 1: play once, 2: play continuous
    private Handler handler = new Handler();

    private PreviewView previewView;
    private ProcessCameraProvider cameraProvider;
    private CameraSelector cameraSelector;
    private boolean isFrontCamera = false;

    private HolisticLandmarkerHelper holisticLandmarkerHelper;
    private MainViewModel mainViewModel;
    private ExecutorService backgroundExecutor;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_predict);

        initializeUI();
        initializeTextToSpeech();
        initializeViewModel();
        checkAndRequestCameraPermission();

        // Receive the landmarks data from the Intent
        String landmarksData = getIntent().getStringExtra("landmarks_data");
        if (landmarksData != null) {
            try {
                JSONArray jsonArray = new JSONArray(landmarksData);
                Log.d(TAG, "Received landmarks: " + "DICKS FOR LIFE");

                // Pass the data to the model predictor
                modelPredictor.predict(jsonArray.toString());
            } catch (JSONException e) {
                Log.e(TAG, "Error parsing landmarks JSON", e);
            }
        }
    }

    // UI Initialization
    private void initializeUI() {
        predictionText = findViewById(R.id.predictionText);
        audioButton = findViewById(R.id.audio_button);
        modelPredictor = new ModelPredictor(this, "fsl_model.tflite", this); // Specify the correct model filename

        // Set up the 'About' button to open AboutScreenActivity
        ImageButton aboutButton = findViewById(R.id.aboutButton);
        aboutButton.setOnClickListener(v -> {
            Intent intent = new Intent(PredictActivity.this, AboutScreenActivity.class);
            startActivity(intent);
        });

        // Camera switch
        ImageButton cameraSwitchButton = findViewById(R.id.cameraSwitch);
        cameraSwitchButton.setOnClickListener(v -> switchCamera());

        audioButton.setOnClickListener(v -> showAudioModeMenu());
    }

    // Text-to-Speech Initialization
    private void initializeTextToSpeech() {
        textToSpeech = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) {
                textToSpeech.setLanguage(new Locale("fil", "PH"));
            }
        });
    }

    // ViewModel Initialization
    private void initializeViewModel() {
        mainViewModel = new ViewModelProvider(this).get(MainViewModel.class);
        holisticLandmarkerHelper = new HolisticLandmarkerHelper(
                mainViewModel.getCurrentMinPoseDetectionConfidence(),
                mainViewModel.getCurrentMinPosePresenceConfidence(),
                mainViewModel.getCurrentMinHandLandmarksConfidence(),
                mainViewModel.getCurrentDelegate(),
                RunningMode.LIVE_STREAM,
                this,
                this
        );
        backgroundExecutor = Executors.newSingleThreadExecutor();
    }

    // Camera Permission Handling
    private void checkAndRequestCameraPermission() {
        if (PermissionsHandler.checkAndRequestCameraPermission(this)) {
            startCamera();
        }
    }

    // Camera Initialization
    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                bindPreview();
            } catch (ExecutionException | InterruptedException e) {
                // Handle any errors (including cancellation) here.
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindPreview() {
        previewView = findViewById(R.id.previewView);
        Preview preview = new Preview.Builder().build();
        ImageAnalysis imageAnalyzer = new ImageAnalysis.Builder()
                .setTargetRotation(previewView.getDisplay().getRotation())
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();
        imageAnalyzer.setAnalyzer(backgroundExecutor, this::detectHolistic);

        cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(isFrontCamera ? CameraSelector.LENS_FACING_FRONT : CameraSelector.LENS_FACING_BACK)
                .build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());
        cameraProvider.bindToLifecycle((LifecycleOwner) this, cameraSelector, preview, imageAnalyzer);
    }

    private void switchCamera() {
        isFrontCamera = !isFrontCamera;
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
            bindPreview();
        }
    }

    private void detectHolistic(ImageProxy imageProxy) {
        holisticLandmarkerHelper.detectLiveStream(
                imageProxy,
                isFrontCamera
        );
    }

    // Text-to-Speech Handling
    private void handleAudioButtonClick() {
        String text = predictionText.getText().toString();
        if (audioMode == 1) {
            textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, null);
        } else if (audioMode == 2) {
            speakOutContinuously(text); // Call a separate method for continuous speech
        }
    }

    private void speakOutContinuously(final String text) {
        textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, "continuous");
        textToSpeech.setOnUtteranceProgressListener(new UtteranceProgressListener() {
            @Override
            public void onStart(String utteranceId) {}

            @Override
            public void onDone(String utteranceId) {
                if ("continuous".equals(utteranceId)) {
                    // Use a Handler to post a delayed runnable
                    handler.postDelayed(() -> speakOutContinuously(text), 500); // Adjust the delay as needed
                }
            }

            @Override
            public void onError(String utteranceId) {}
        });
    }

    private void showAudioModeMenu() {
        PopupMenu popupMenu = new PopupMenu(this, audioButton);
        popupMenu.getMenuInflater().inflate(R.menu.audio_mode_menu, popupMenu.getMenu());

        popupMenu.setOnMenuItemClickListener(item -> {
            int itemId = item.getItemId();
            String text = predictionText.getText().toString();
            if (itemId == R.id.audio_off) {
                audioMode = 0;
                audioButton.setImageResource(R.drawable.audio_off);
                textToSpeech.stop(); // Stop any ongoing speech
                Toast.makeText(this, "Audio Mode: Off", Toast.LENGTH_SHORT).show();
            } else if (itemId == R.id.audio_play_once) {
                audioMode = 1;
                audioButton.setImageResource(R.drawable.audio_play_once);
                Toast.makeText(this, "Audio Mode: Play Once", Toast.LENGTH_SHORT).show();
                textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, null); // Read text instantly
            } else if (itemId == R.id.audio_play_continuous) {
                audioMode = 2;
                audioButton.setImageResource(R.drawable.audio_play_continuous);
                Toast.makeText(this, "Audio Mode: Play Continuous", Toast.LENGTH_SHORT).show();
                speakOutContinuously(text); // Read text continuously instantly
            }
            return true;
        });

        popupMenu.show();
    }

    // Lifecycle Methods
    @Override
    protected void onDestroy() {
        if (textToSpeech != null) {
            textToSpeech.stop();
            textToSpeech.shutdown();
        }
        if (backgroundExecutor != null) {
            backgroundExecutor.shutdown();
        }
        super.onDestroy();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PermissionsHandler.CAMERA_PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Camera permission granted, start the camera
                startCamera();
            } else {
                Log.d("PredictActivity", "Camera permission denied");
            }
        }
    }



    @Override
    public void onResults(HolisticLandmarkerHelper.ResultBundle resultBundle) {


        // Process the landmarkBuffer as needed
    }

  /*  void receiveLandmarks(String landmarks) throws JSONException {
        ModelPredictor modelPredictor1 = new ModelPredictor(this, "fsl_model.tflite", this);
        Log.d(TAG, modelPredictor1.toString());

        try {
            // Wrap the string if necessary
            if (!landmarks.startsWith("[") || !landmarks.endsWith("]")) {
                landmarks = "[" + landmarks + "]";
            }

            JSONArray jsonArray = new JSONArray(landmarks);
            Log.d(TAG, String.valueOf(jsonArray));
            modelPredictor1.predict(String.valueOf(jsonArray));
        } catch (JSONException e) {
            Log.e(TAG, "Error parsing landmarks JSON", e);
        }
    }*/

    @Override
    public void onError(String error, int errorCode) {
        // Handle the error here
        Log.e(TAG, "Holistic Landmarker error: " + error);
        runOnUiThread(() -> Toast.makeText(this, error, Toast.LENGTH_SHORT).show());
    }

    @Override
    public void onPrediction(String action) {
        Log.d(TAG, "onPrediction called with action: " + action);
        runOnUiThread(() -> predictionText.setText(action));
    }
}