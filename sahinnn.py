import cv2
import numpy as np

def enhance_low_light(image_path: str, output_path: str = "enhanced_output.jpg") -> None:
    # ছবি লোড করুন
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Cannot load image from: {image_path}")
    
    # গ্রেস্কেল এ কনভার্ট করে আলো-তিমির অঞ্চল শনাক্তের জন্য
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Step 1: লো-লাইট এলাকা শনাক্ত করার জন্য থ্রেশহোল্ডিং (80 এর নিচের পিক্সেলগুলো লো-লাইট)
    threshold_value = 80
    _, low_light_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Step 2: হিস্টোগ্রাম ইকুয়ালাইজেশন দিয়ে গ্রেস্কেল ইমেজের কনট্রাস্ট বাড়ানো
    hist_eq = cv2.equalizeHist(gray)

    # Step 3: অ্যাডাপ্টিভ স্পেশাল ফিল্টারিং (গাউসিয়ান ব্লার দিয়ে ছবি একটু স্মুথ করা)
    smoothed = cv2.GaussianBlur(hist_eq, (5, 5), 0)

    # Step 4: শার্পেনিং (লাপ্লাসিয়ান দিয়ে ছবি আরো তীক্ষ্ণ করা)
    laplacian = cv2.Laplacian(smoothed, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    sharpened = cv2.addWeighted(smoothed, 1.5, laplacian, -0.5, 0)

    # Step 5: শুধুমাত্র লো-লাইট অংশে উন্নত ছবি ব্যবহার করে মুল ছবির সাথে মিশিয়ে দেওয়া
    enhanced_gray = np.where(low_light_mask == 255, sharpened, gray)

    # গ্রেস্কেলকে আবার BGR এ কনভার্ট করা
    enhanced_gray_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    # তিন চ্যানেল মাস্ক তৈরি করে শুধুমাত্র লো-লাইট অংশগুলো পরিবর্তন করা
    mask_3ch = cv2.merge([low_light_mask] * 3)
    result = np.where(mask_3ch == 255, enhanced_gray_bgr, original)

    # ফলাফল সেভ করা
    cv2.imwrite(output_path, result)
    print(f"[✓] Enhanced image saved to: {output_path}")

# রান করার জন্য উদাহরণ
if __name__ == "__main__":
    enhance_low_light("your_photo.jpg")
