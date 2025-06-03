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

    // ��k�G�]�w��
    void setData(int value);

    // ��k�G���o��
    int getData() const;

    // ��ܸ��
    void display() const;

    // ======= Leetcode Solutions =======
    std::vector<int> Leetcode_Sol_xxx(std::vector<int>& numbers, int target, int _solution);
    std::vector<int> Exam_xxx(std::vector<int>& numbers, int target);
    // ======= Leetcode Solutions =======

};



extern DLL_API MyDP MyDPInstance;
