Giới thiệu về ML:
Phân loại ML:

Các khái niệm:
    -- Overfit (low bias high variance) khi model quá khớp với dữ liệu training
    -- Underfit (high bias low variance) khi model quá sai lệch
    --> Bias-Variance trade-off

MSE (mean squared error) kết hợp cả sai số và phương sai

![alt mse](./images/mse.png)

Giới thiệu các cách để giảm thiểu overfit
Các hàm đa thức bậc cao sẽ rất phù hợp với dữ liệu train nhưng sẽ dễ gây ra overfit
    
![alt high order polynomial function](./images/high-order-function.png)

+ Với dữ liệu đủ lớn chia ra 3 subset: train-set, validate-set, test-set

+ Cross-validation: thường dùng khi không có dữ liệu đủ lớn, không thể chia ra nhiều các subset, dùng để giảm thiểu overfit
    Có 2 loại chính:
        
        -- exhaustive (toàn diện)
        Chia ra n subset (với n là độ lớn dữ liệu), và n lần học. Mỗi lần học thứ i sẽ tách phần tử thứ i làm test-set
        -> Leave-One-Out-Cross-Validation (LOOCV)

        -- non-exhaustive (Không toàn diện)
        Tương tự LOOCV nhưng dữ liệu được chia thành K phần lớn hơn (k thường là 3, 5, 10)
        -> K-fold cross-validation

        Cũng có thể random test-set từ training-set
        -> holdout

+ Regularization: thay đổi mô hình hoặc cách chạy để làm giảm overfit
    -- Early stoping: ép mô hình chạy trong khoảng thời gian hoặc đặt một số tiêu chí dừng.
    -- regularization loss function: Thêm vào hàm mất mát một số hạng nào đó để đánh giá độ phức tạp của mô hình.

+ Lựa chọn thuộc tính và giảm số chiều:
    -- với quá nhiều thuộc tính sẽ làm tăng số lần thực hiện tính toán
    -- Giảm số chiều có thể sẽ làm mất mát thông tin.

    