#pragma once

#define Build_DLL

#ifdef Build_DLL 
#define DLL_API _declspec(dllexport)
#else 
#define DLL_API _declspec(dllimport)
#endif // BuildDLL _declspec(DLLExport)

#include <string>
#include <iostream>
#include <vector>       // �Y�����ܼƥΨ� std::vector
#include <map>          // �Y�� std::map �ܼ�
#include <unordered_map>// �Y�� std::unordered_map �ܼ�

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
    int* data;  // ���]��Ʀ����O����

public:
    // �w�]�غc��
    MyDP();

    // �a�Ѽƫغc��
    MyDP(int value);

    // �Ѻc��
    ~MyDP();

    //�ƦC(P)�Gn�Ӥ��P���F��ƦC�A�����Gn!
    int Premutation(int n);

    //P(n, k)�Gn�Ӥ��P���F���k�ӱƦC�A�����GP(n, k) = n!/ (n - k)
    static int Premutation_pick_k(int n, int k);

    //�զX(C)�Gn�Ӥ��P���F���k�ӲզX�A�����GC(n, k) = n!/ ((n - k)!*k!)
    static int Combination(int n, int k);

    //�T�B���ƲզX(H)�Gn�Ӥ��P���F���k�ӲզX(�i���ƨ�)�A�����GH(n, k) = C(n + k - 1, k) = C(n + k - 1, n - 1)
    static int Combination_H(int n = 8, int k = 5);

    // ��k�G�]�w��
    void setData(int value);

    // ��k�G���o��
    int getData() const;

    // ��ܸ��
    void display() const;

    //�x�}���� (���j)
    MyMatrix matrix_pow(MyMatrix& A, int n);
    //�x�}���� (����iterator)
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
    //�ۥX�D
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
