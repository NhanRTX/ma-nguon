cách sử dụng code và model
dữ liệu cần chuẩn bị : tải code, file CNNmodel,YOLOv8 về máy
b1  : mở colab và chạy
!pip install ultralytics
!pip install gradio
b2: dán up code vào colab tải CNNmodel và YOLOv8 lên colab
b3: dán file CNN vào /content/CNNmodel.tflite trong cnn_interpreter = tf.lite.Interpreter(model_path="/content/CNNmodel.tflite"
b4: dán file YOLO vào/content/YOLOv8.pt trong detector = YOLO("/content/YOLOv8.pt")
b5: chạy và sử dụng
