import torch
import cv2
import numpy as np
import keyboard

# Counting values for Hi-Lo strategy
COUNTING_VALUES = {
    '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,    # Low cards (+1)
    '7': 0, '8': 0, '9': 0,                     # Neutral cards (0)
    '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1  # High cards (-1)
}

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced

def run_card_counter():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='data/blackjack.pt')
    model.conf = 0.7  # Increased confidence threshold
    if torch.cuda.is_available():
        model.cuda()
    
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    running_count = 0
    seen_cards = set()
    
    print("Starting detection... Press 'q' to quit, 'r' to reset count")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = preprocess_frame(frame)
        results = model(processed_frame)
        
        if len(results.xyxy[0]) > 0:
            boxes = results.xyxy[0].cpu().numpy()
            for box in boxes:
                if box[4] > 0.7:  # Increased confidence threshold here too
                    card_name = model.names[int(box[5])]
                    if card_name not in seen_cards:
                        rank = card_name[:-1]
                        count_change = COUNTING_VALUES.get(rank, 0)
                        running_count += count_change
                        seen_cards.add(card_name)
                        if count_change != 0:
                            print(f"Detected: {card_name}, Count Change: {count_change:+d}, Total: {running_count}")
                        else:
                            print(f"Detected: {card_name}, Neutral card, Count remains: {running_count}")
        
        annotated_frame = results.render()[0]
        cv2.putText(annotated_frame, f'Count: {running_count}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Cards: {", ".join(sorted(seen_cards))}', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Card Counter', annotated_frame)
        
        # Use OpenCV's waitKey instead of keyboard library
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            running_count = 0
            seen_cards.clear()
            print("\nCount reset!")
            print("Starting fresh detection...")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_card_counter()