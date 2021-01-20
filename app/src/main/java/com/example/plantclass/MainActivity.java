package com.example.plantclass;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.ProgressDialog;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;


import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.ml.common.modeldownload.FirebaseModelDownloadConditions;
import com.google.firebase.ml.common.modeldownload.FirebaseModelManager;
import com.google.firebase.ml.custom.FirebaseCustomRemoteModel;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.lang.reflect.Array;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.ProtocolException;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    static final int REQUEST_IMAGE_CAPTURE = 1;
    private static final String TAG = "MainActivity";
    private ImageView imageView;
    private TextView textView;
    private Bitmap imageBitmap;
    private String upLoadServerUri = "http://192.168.56.1/hack/phpFile.phpp";;
    private String[] CornDet2Classes = {"Leaf spot", "Common rust", "Healthy", "Leaf blight"};
    private int CornDet2ClassNumber = 4;
    Interpreter interpreter;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        textView = findViewById(R.id.textView);
    }

    public void dispatchTakePictureIntent(View view) {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        try {
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
        } catch (ActivityNotFoundException e) {
            Toast.makeText(this, "Error: " + e.getMessage(), Toast.LENGTH_SHORT).show();
            Log.e(TAG, "dispatchTakePictureIntent: " + e.getMessage() );
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            Bundle extras = data.getExtras();
            imageBitmap = (Bitmap) extras.get("data");
            imageView.setImageBitmap(imageBitmap);
            textView.setText("Image Set");
        }
    }

    public void upload(View view) {
        if(imageBitmap==null){
            textView.setText("First Select Image, then try to upload");
            return;
        }
        ProgressDialog pd = new ProgressDialog(this);
        pd.setCancelable(false);
        pd.setMessage("Uploading, please wait...");
        pd.setProgressStyle(ProgressDialog.STYLE_HORIZONTAL);
        pd.setProgress(0);
        pd.setMax(100);
        pd.show();

        process("CornDet2", CornDet2Classes, CornDet2ClassNumber );

        pd.dismiss();
    }

    private void process(String modelName, String[] classesName, int classesNumber) {
        FirebaseCustomRemoteModel remoteModel =
                new FirebaseCustomRemoteModel.Builder(modelName).build();
        FirebaseModelDownloadConditions conditions = new FirebaseModelDownloadConditions.Builder()
                .requireWifi()
                .build();
        FirebaseModelManager.getInstance().download(remoteModel, conditions)
                .addOnSuccessListener(new OnSuccessListener<Void>() {
                    @Override
                    public void onSuccess(Void v) {
                        FirebaseCustomRemoteModel remoteModel = new FirebaseCustomRemoteModel.Builder(modelName).build();
                        FirebaseModelManager.getInstance().getLatestModelFile(remoteModel)
                                .addOnCompleteListener(new OnCompleteListener<File>() {
                                    @Override
                                    public void onComplete(@NonNull Task<File> task) {
                                        File modelFile = task.getResult();
                                        if (modelFile != null) {
                                            interpreter = new Interpreter(modelFile);
                                            get(classesName,classesNumber);
                                        }
                                    }
                                });
                    }
                });
    }

    private void get(String[] classesName, int classesNumber){
        int shape1 = 256;
        Bitmap bitmap = Bitmap.createScaledBitmap(imageBitmap, shape1, shape1, true);
        ByteBuffer input = ByteBuffer.allocateDirect(shape1 * shape1 * 3 * 4).order(ByteOrder.nativeOrder());
        for (int y = 0; y < shape1; y++) {
            for (int x = 0; x < shape1; x++) {
                int px = bitmap.getPixel(x, y);

                // Get channel values from the pixel value.
                int r = Color.red(px);
                int g = Color.green(px);
                int b = Color.blue(px);

                // Normalize channel values to [-1.0, 1.0]. This requirement depends
                // on the model. For example, some models might require values to be
                // normalized to the range [0.0, 1.0] instead.
                float rf = (r - 127) / 255.0f;
                float gf = (g - 127) / 255.0f;
                float bf = (b - 127) / 255.0f;

                input.putFloat(rf);
                input.putFloat(gf);
                input.putFloat(bf);
            }
        }
        int bufferSize = classesNumber * java.lang.Float.SIZE / java.lang.Byte.SIZE;
        ByteBuffer modelOutput = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());
        interpreter.run(input, modelOutput);

        modelOutput.rewind();
        FloatBuffer probabilities = modelOutput.asFloatBuffer();
        String st = "";
        float[] iList = new float[probabilities.capacity()];
        for (int i = 0; i < probabilities.capacity(); i++) {
            st += " | " + probabilities.get(i);
            iList[i] = probabilities.get(i);

        }

        String name = classesName[getIndexOfMax(iList)];
        Log.i(TAG, name);

        textView.setText("Disease Name is " + name);
    }

    public int getIndexOfMax(float array[]) {
        if (array.length == 0) {
            return -1; // array contains no elements
        }
        float max = array[0];
        int pos = 0;

        for(int i=1; i<array.length; i++) {
            if (max < array[i]) {
                pos = i;
                max = array[i];
            }
        }
        return pos;
    }

}