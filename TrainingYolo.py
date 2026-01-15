from ultralytics import YOLO

def main():
    model = YOLO('yolo11n-cls.pt') 

    results = model.train(
        data=r'C:\Users\ROG\Desktop\Capstone\dataset',
        epochs=50, 
        imgsz=224,              
        device=0,   
        batch=32,
        degrees=15.0,    
        fliplr=0.5,      
        hsv_v=0.4,
        project='DriverDrowsiness',
        name='yolo11n_training',
    )

    best_model = YOLO('DriverDrowsiness/yolo11n_training/weights/best.pt')
    best_model.export(format='ncnn')

if __name__ == '__main__':
    main()