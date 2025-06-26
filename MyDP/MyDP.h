#pragma once

#define Build_DLL

#ifdef Build_DLL 
#define DLL_API _declspec(dllexport)
#else 
#define DLL_API _declspec(dllimport)
#endif // BuildDLL _declspec(DLLExport)

#include <string>
#include <iostream>
#include <vector>       // 若成員變數用到 std::vector
#include <map>          // 若有 std::map 變數
#include <unordered_map>// 若有 std::unordered_map 變數

class MyMatrix {
public:
    int Matrix_size = 2;
    std::vector<std::vector<int>> mat;
    MyMatrix();
    MyMatrix(int size);
    MyMatrix(const std::vector<std::vector<int>>& init);

    MyMatrix operator* (const MyMatrix& other) const;
    static MyMatrix identity(int size);
    ~MyMatrix();
};

class MyDP {
private:
    int* data;  // 假設資料成員是指標

public:
    // 預設建構式
    MyDP();

    // 帶參數建構式
    MyDP(int value);

    // 解構式
    ~MyDP();

    //排列(P)：n個不同的東西排列，公式：n!
    int Premutation(int n);

    //P(n, k)：n個不同的東西取k個排列，公式：P(n, k) = n!/ (n - k)
    static int Premutation_pick_k(int n, int k);

    //組合(C)：n個不同的東西取k個組合，公式：C(n, k) = n!/ ((n - k)!*k!)
    static int Combination(int n, int k);

    //三、重複組合(H)：n個不同的東西取k個組合(可重複取)，公式：H(n, k) = C(n + k - 1, k) = C(n + k - 1, n - 1)
    static int Combination_H(int n = 8, int k = 5);

    // 方法：設定值
    void setData(int value);

    // 方法：取得值
    int getData() const;

    // 顯示資料
    void display() const;

    //矩陣次方 (遞迴)
    MyMatrix matrix_pow(MyMatrix& A, int n);
    //矩陣次方 (遞推iterator)
    MyMatrix matrix_exp_iterative(MyMatrix base, int exp);


    // ======= Leetcode Solutions =======
    std::vector<int> Leetcode_Sol_xxx(std::vector<int>& numbers, int target, int _solution);
    std::vector<int> Exam_xxx(std::vector<int>& numbers, int target);

    //Fibonacci Number
    long long Leetcode_Sol_509(int n);
    long long fib(int n); 
    //Tribonacci Number
    long long Leetcode_Sol_1137(int n);
    long long tribonacci(int n);

    int Leetcode_Sol_312(std::vector<int>& nums,int _solution);
    int bottom_up_iterator(std::vector<int>& nums);
    int Top_dowm_Memoization(std::vector<int>& nums);
    //自出題
    std::vector<int> Leetcode_Sol_312_bonus(std::vector<int>& nums, int _solution);
    std::vector<int> bottom_up_iterator_312_bonus(std::vector<int>& nums);
    void preorder_312(const std::vector<std::vector<int>>& pick, int left, int right, std::vector<int>& order);
    std::vector<int> Top_dowm_Memoization_312_bonus(std::vector<int>& nums);

    std::string Leetcode_Sol_943(std::vector<std::string>& words,int _solution);
    std::string DP_BitMask_TSP_943(std::vector<std::string>& words);

    int Leetcode_Sol_1012(int n);
    // ======= Leetcode Solutions =======

};



extern DLL_API MyDP MyDPInstance;
