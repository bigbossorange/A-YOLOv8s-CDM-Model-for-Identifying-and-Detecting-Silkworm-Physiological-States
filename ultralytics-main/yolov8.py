import sys

sys.path.append("autodl-tmp/yolov8/")

from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    # 直接使用预训练模型创建模型
    # model = YOLO('yolov8n.pt')
    # model.train(**{'cfg':'ultralytics/cfg/default.yaml', 'data':'ultralytics/models/yolo/detect/mydata/traffic.yaml'}, epochs=10, imgsz=640, batch=32)

    # #使用yaml配置文件来创建模型，并导入预训练权重
     model = YOLO('yolov8.yaml')
     model.load('yolov8s.pt')
     model.train(**{
         'cfg':'ultralytics/cfg/default.yaml', 
         'data':'coco.yaml',
         'lr0': 0.01,
         'lrf': 0.01,
         'weight': 0.005,
         'momentum': 0.937,
     }, epochs=200, imgsz=640, batch=16, name='train-ContextAggregation')

    # #     # 模型验证：用验证集
    #     model = YOLO('runs/detect/train-ContextAggregation/weights/best.pt')
    #     model.val(**{'data':'ultralytics/models/yolo/detect/mydata/traffic.yaml', 'name':'val-ContextAggregation', 'batch':32}) #模型验证用验证集
    #     model.val(**{'data':'ultralytics/models/yolo/detect/mydata/traffic.yaml', 'split':'test', 'iou':0.9}) #模型验证用测试集

    #     # 模型推理：
    #model = YOLO('runs/detect/train-ContextAggregation/weights/best.pt')
    #model.predict(source='ultralytics/assets', name='predict-ContextAggregation',
                #  **{'save': True})  # 若没有save:True，则会找不到保存路径


    #yolo predict model='D:/yolov8/ultralytics-main/runs/detect/fire3/weights/best.pt' source='ultralytics/assets/12.jpg'
#    result = detect_pointer_background_color('ultralytics/assets/12.jpg', 'D:/yolov8/ultralytics-main/runs/detect/fire3/weights/best.pt')