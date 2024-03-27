from ultralytics import YOLO
if __name__=='__main__':
    # Create a new YOLO model from scratch
    model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')

    # Load a pretrained YOLO model (recommended for training)
    model.load('weights/yolov8n.pt')

    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(data='VOC.yaml', epochs=200,batch=64,seed=42, device='1',**{'cfg':'ultralytics/cfg/default.yaml'})

    # Evaluate the model's performance on the validation set
    # results = model.val()
    #
    # # Export the model to ONNX format
    # success = model.export(format='onnx', opset=12,imgsz=320)


