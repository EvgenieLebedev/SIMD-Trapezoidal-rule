#include <immintrin.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <locale>

alignas(32) float x_vals_arr[8]; // Глобальный массив для выравненных данных

// Функция, которую будем интегрировать
float func(float x) {
    return std::sin(x);
}

double func_double(double x) {
    return std::sin(x);
}

// Метод прямоугольников с использованием SIMD
float rectangle_method_simd(float (*f)(float), float a, float b, int n) {
    float delta_x = (b - a) / n;
    __m256 integral_vec = _mm256_setzero_ps();

    int simd_width = 8;
    int simd_end = n - (n % simd_width);

    for (int i = 0; i < simd_end; i += simd_width) {
        __m256 indices = _mm256_set_ps(i + 7, i + 6, i + 5, i + 4, i + 3, i + 2, i + 1, i);
        __m256 x_vals = _mm256_add_ps(_mm256_mul_ps(indices, _mm256_set1_ps(delta_x)), _mm256_set1_ps(a));

        _mm256_store_ps(x_vals_arr, x_vals); // Сохраняем x_vals в выровненный массив

        __m256 f_vals = _mm256_set_ps(f(x_vals_arr[7]), f(x_vals_arr[6]), f(x_vals_arr[5]), f(x_vals_arr[4]),
            f(x_vals_arr[3]), f(x_vals_arr[2]), f(x_vals_arr[1]), f(x_vals_arr[0]));
        integral_vec = _mm256_add_ps(integral_vec, f_vals);
    }

    alignas(32) float integral_arr[8];
    _mm256_store_ps(integral_arr, integral_vec); // Выровненное хранилище результата

    float integral = 0;
    for (int i = 0; i < 8; ++i) {
        integral += integral_arr[i];
    }

    for (int i = simd_end; i < n; ++i) {
        integral += f(a + i * delta_x);
    }

    return integral * delta_x;
}

// Обычный метод прямоугольников
double rectangle_method(double (*f)(double), double a, double b, int n) {
    double delta_x = (b - a) / n;
    double integral = 0;

    for (int i = 0; i < n; i++) {
        double x_i = a + i * delta_x;
        integral += f(x_i);
    }

    integral *= delta_x;
    return integral;
}

int main() {
    setlocale(LC_ALL, "Russian");
    const int n = 10000000; // Количество разбиений

    // Обычный метод
    auto start1 = std::chrono::high_resolution_clock::now();
    double result1 = rectangle_method(func_double, 0.0, 3.14159, n);
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration1 = end1 - start1;

    // Метод с SIMD
    auto start2 = std::chrono::high_resolution_clock::now();
    float result2 = rectangle_method_simd(func, 0.0f, 3.14159f, n);
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration2 = end2 - start2;

    // Вывод результатов
    std::cout << "Обычный метод: " << result1 << " за " << duration1.count() << " секунд\n";
    std::cout << "SIMD метод: " << result2 << " за " << duration2.count() << " секунд\n";

    return 0;
}
