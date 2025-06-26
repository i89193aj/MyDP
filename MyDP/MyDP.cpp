/*Oscar MyDP*/

#include <iostream>
#include <cstdlib>  // rand(), srand()
#include <ctime>    // time()
#include <random>   // C++11 亂數庫
#include <vector>   // std::vector
#include <numeric>  // std::iota
#include <map>      //std::map
#include <unordered_map>  //std::unordered_map
#include <unordered_set>  //std::unordered_set
#include <queue>  //std::deque

#include"MyDP.h"
using namespace std;

DLL_API MyDP MyDPInstance;
enum LeetcodeExam {
    Leetcodexxx,
    Leetcode509,
    Leetcode1137,
    Leetcode312,
    Leetcode943,
    Leetcode1012,

    None,
};


bool wordBreak(string s, vector<string>& wordDict) {
    unordered_set<string> dict(wordDict.begin(), wordDict.end());
    int n = s.size();
    vector<bool> dp(n + 1, false);
    dp[0] = true;

    for (int i = 1; i <= n; ++i) {
        for (int j = 0; j < i; ++j) {
            if (dp[j] && dict.count(s.substr(j, i - j))) {
                dp[i] = true;
                break;
            }
        }
    }
    return dp[n];
}

    


int main()
{
    #pragma region temporary_test
    vector<string> sd = { "leet","leetc","code","ode" };
    vector<string> sdd = { "catg","ctaagt","gcta","ttca","atgcatc" };
    vector<int> test = { 1,2 };
    string sss = "leetcode";
    bool aa = wordBreak(sss, sd);


    #pragma endregion


    //try case
    LeetcodeExam ExamEnum = Leetcode943;    //ChangeForExam
    //intput
    vector<int> vInput1 = { 3,8,5};              
    vector<int> vInput2 = { 7,13,11,10,1 };
    vector<string> vInput_String = { "catg","ctaagt","gcta","ttca","atgcatc" };
    vector<vector<int>> vvInput1 = { {1,2} ,{2,3},{3,4},{1,3} };
    string strinput1 = "bab";
    string strinput2 = "xaabacxcabaaxcabaax";
    int iInput1 = 0;int iInput2 = 0;
    //output
    int Ans = 0; vector<int> AnsVector; string AnsStr = "";bool Ansbool = false;
    long long l_Ans = 0;
    MyDP* Implementation = new MyDP();

    switch (ExamEnum)
    {
    case Leetcode509:   //Fibonacci Number：0, 1, 1, 2, 3, 5, 8, 13, ...
        l_Ans = Implementation->Leetcode_Sol_509(3);
        l_Ans = Implementation->Leetcode_Sol_509(4);
        l_Ans = Implementation->Leetcode_Sol_509(5);
        l_Ans = Implementation->Leetcode_Sol_509(6);
        l_Ans = Implementation->Leetcode_Sol_509(7);
        break;
    case Leetcode1137:  //Tribonacci Number：0, 1 ,1, 2, 4, 7, 13, 24, ...
        l_Ans = Implementation->Leetcode_Sol_1137(3);
        l_Ans = Implementation->Leetcode_Sol_1137(4);
        l_Ans = Implementation->Leetcode_Sol_1137(5);
        l_Ans = Implementation->Leetcode_Sol_1137(6);
        l_Ans = Implementation->Leetcode_Sol_1137(7);
        break;

    case Leetcode312:  //Tribonacci Number：0, 1 ,1, 2, 4, 7, 13, 24, ...
        Ans = Implementation->Leetcode_Sol_312(vInput1,1);
        AnsVector = Implementation->Leetcode_Sol_312_bonus(vInput1, 1);
        break;
    case Leetcode943:  //Tribonacci Number：0, 1 ,1, 2, 4, 7, 13, 24, ...
        AnsStr = Implementation->Leetcode_Sol_943(vInput_String, 1);
    case Leetcode1012:  //Tribonacci Number：0, 1 ,1, 2, 4, 7, 13, 24, ...
        //Ans = Implementation->Leetcode_Sol_1012(iInput1);
        Ans = Implementation->Leetcode_Sol_1012(109);
        Ans = Implementation->Leetcode_Sol_1012(20);
        Ans = Implementation->Leetcode_Sol_1012(100);
        Ans = Implementation->Leetcode_Sol_1012(1000);
        Ans = Implementation->Leetcode_Sol_1012(3446);
        break;
    default:
        break;
    }
    #pragma region MyDP
    MyDP obj1;              // 呼叫預設建構式
    obj1.display();

    MyDP obj2(10);          // 呼叫帶參數建構式
    obj2.display();

    obj1.setData(20);                // 修改資料成員
    obj1.display();

    return 0;
    #pragma endregion

    
}

#pragma region MyMatrix
MyMatrix::MyMatrix() {
    mat = vector<vector<int>>(Matrix_size, vector<int>(Matrix_size, 0));
} 

MyMatrix::MyMatrix(int size)
    : Matrix_size(size), mat(vector<vector<int>>(size, vector<int>(size, 0))) {
}

MyMatrix::MyMatrix(const std::vector<std::vector<int>>& init):Matrix_size(init.size()), mat(init){}

/*做{{1,0},{0,1}}恆等轉換。*/
MyMatrix MyMatrix::identity(int size) {
    vector<vector<int>> id(size, vector<int>(size, 0));
    for (int i = 0; i < size; ++i)
        id[i][i] = 1;
    return MyMatrix(id);
}
/*左右值複習：
1.只能傳入右值，不能傳入左值(有名字的) => 參考(少資源) + 一次move(最省，但不能用左值代入)
MyMatrix::MyMatrix(std::vector<std::vector<int>>&& init) : Matrix_size(init.size()), mat(std::move(init)){}
ex：
std::vector<std::vector<int>> tmp = {{1, 1}, {1, 0}};
MyMatrix A(tmp);                // ❌ 編譯錯誤：沒有對應的建構子
MyMatrix A({{1, 1}, {1, 0}});   // ✅ 可以

2.可以傳入左右值，但是傳進來是以copy的方式，不是參考 => 一次copy + 一次move
MyMatrix(std::vector<std::vector<int>> init) : Matrix_size(init.size()), mat(std::move(init)) {}

3.傳入參考值，但由於受限const，所以不能移動原本位置(容器)的值 => 因為移動代表就改變原本的container => 參考(少資源) + copy to mat
Matrix::MyMatrix(const std::vector<std::vector<int>>& init):Matrix_size(init.size()), mat(init){}
*/



MyMatrix::~MyMatrix() {}

MyMatrix MyMatrix::operator* (const MyMatrix& other) const {
    int size = other.Matrix_size;
    MyMatrix result(size);
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            for (int k = 0; k < size; ++k)
                result.mat[i][j] += mat[i][k] * other.mat[k][j];
    return result;
}
#pragma endregion


#pragma region MyDP
// 預設建構式，初始化指標
MyDP::MyDP() : data(new int(0)) {
    std::cout << "Default constructor called. Data initialized to 0." << std::endl;
}

// 帶參數建構式，初始化指標並設置初始值
MyDP::MyDP(int value) : data(new int(value)) {
    std::cout << "Parameterized constructor called. Data initialized to " << value << "." << std::endl;
}

// 解構式，釋放動態分配的記憶體
MyDP::~MyDP() {
    delete data;
    std::cout << "Destructor called. Memory for data released." << std::endl;
}

// 設定資料成員
void MyDP::setData(int value) {
    *data = value;
}

// 取得資料成員
int MyDP::getData() const {
    return *data;
}

// 顯示資料
void MyDP::display() const {
    std::cout << "Data: " << *data << std::endl;
}

MyMatrix MyDP::matrix_pow(MyMatrix& A, int n) {
    if (n == 1) return A;
    MyMatrix half = matrix_pow(A, n / 2);
    MyMatrix result = half * half;
    if (n % 2 == 1) result = result * A;
    return result;
}

MyMatrix MyDP::matrix_exp_iterative(MyMatrix base, int exp) {
    MyMatrix result = MyMatrix::identity(base.Matrix_size);
    while (exp > 0) {
        if (exp % 2 == 1) result = result * base;
        base = base * base;
        exp /= 2;
    }
    return result;
}

#pragma region MyRegion
//排列(P)：
// n!：n個不同的東西排列
int MyDP::Premutation(int n) {
    int res = 1;
    for (int i = 1; i <= n; i++)
        res *= i;

    return res;
}

//P(n, k)：n個不同的東西取k個排列
//公式：P(n, k) = n!/ (n - k)
int MyDP::Premutation_pick_k(int n, int k) {
    int res = 1;
    for (int i = 0; i < k; i++)
        res *= n - i;
    return res;
}

//組合(C)：
//C(n, k)：n個不同的東西取k個組合
//公式：C(n, k) = n!/ ((n - k)!*k!)
int MyDP::Combination(int n, int k) {
    long long res = 1;
    k = min(n - k, k);
    for (int i = 1; i <= k; i++)
        res = res * (n - i + 1) / i;
    return res;
}

//三、重複組合(H)：
//H(n, k)：n個不同的東西取k個組合(可重複取)
//公式：H(n, k) = C(n + k - 1, k) = C(n + k - 1, n - 1)
int MyDP::Combination_H(int n, int k) {
    int a = n + k - 1;
    return MyDP::Combination(a, k);
}

#pragma endregion


#pragma region Fibonacci Number
long long MyDP::Leetcode_Sol_509(int n) {
    return MyDP::fib(n);
}
long long MyDP::fib(int n) {
    if (n <= 1) return n;
    MyMatrix base({ {1, 1}, {1, 0} });
    MyMatrix result = matrix_pow(base, n - 1);  //Recursion S(logn)             
    if (false)
        MyMatrix result = matrix_exp_iterative(base, n - 1); //iterator S(1)
    return result.mat[0][0]; // F(n)
}
#pragma endregion

#pragma region Tribonacci Number
long long MyDP::Leetcode_Sol_1137(int n) {
    return MyDP::tribonacci(n);
}

long long MyDP::tribonacci(int n) {
    if (n == 0) return 0;
    if (n == 1 || n == 2) return 1;
    /*  | T(n)   |     =   | 1 1 1 |   * | T(n-1) |
        | T(n-1) |         | 1 0 0 |     | T(n-2) |
        | T(n-2) |         | 0 1 0 |     | T(n-3) |   */
    MyMatrix tranfer({ {1,1,1} ,{1,0,0} ,{0,1,0} });
    //需要注意這裡轉幾次，才是base case => 3*3的矩陣，需要 n - 2 (因為基底3個：T0 = 0, T1 = 1, T2 = 1)
    MyMatrix result = matrix_pow(tranfer, n - (tranfer.Matrix_size - 1));   
    vector<int> X({ 1,1,0 });
    int ans = 0;
    for (int c = 0; c < X.size(); c++) {
        ans += result.mat[0][c] * X[c];
    }
    return ans;
}
#pragma endregion

#pragma region Burst Balloons
int MyDP::Leetcode_Sol_312(vector<int>& nums,int _solution){
    switch (_solution)
    {
    case 1:
        return bottom_up_iterator(nums);
    case 2:
        return Top_dowm_Memoization(nums);
    default:
        return -1; // 確保所有路徑都有回傳值
    }
}

int MyDP::bottom_up_iterator(vector<int>& nums) {
    int n = nums.size() + 2;//先補足最左、右邊的1
    //初始化
    vector<vector<int>> dp(n, vector<int>(n, 0));//dp[i][j]：[i~j]在這個區間得到的最多分數 (不選：i、j)
    vector<int> balloons(n, 1);
    for (int i = 0; i < nums.size(); i++)
        balloons[i + 1] = nums[i];

    //for + dp func：
    for (int len = 2; len < n; len++) { //len = 2開始 => 左右各有1
        int interval_end = n - len;
        for (int start = 0; start < interval_end; start++) {
            int end = start + len;
            for (int pick = start + 1; pick < end; pick++) {
                dp[start][end] = max(dp[start][end],
                    dp[start][pick] + dp[pick][end] + balloons[start] * balloons[pick] * balloons[end]);
            }
        }

    }
    return dp[0][n - 1];
}

int MyDP::Top_dowm_Memoization(vector<int>& nums) {
    return 0;
}

vector<int> MyDP::Leetcode_Sol_312_bonus(vector<int>& nums, int _solution) {
    switch (_solution)
    {
    case 1:
        return bottom_up_iterator_312_bonus(nums);
    case 2:
        return Top_dowm_Memoization_312_bonus(nums);
    default:
        return {}; // 確保所有路徑都有回傳值
    }
}

vector<int> MyDP::bottom_up_iterator_312_bonus(vector<int>& nums) {
    int n = nums.size() + 2;    //把左右兩邊的1補齊
    vector<int> balloons(n + 2, 1);
    for (int i = 0; i < nums.size(); ++i) 
        balloons[i + 1] = nums[i];

    vector<vector<int>> dp(n, vector<int>(n, 0));
    vector<vector<int>> picks(n, vector<int>(n, -1));  // picks[i][j] = 最佳決策 k
    unordered_set<int> freq;
    std::deque<int> pick2;
    for (int len = 2; len < n; ++len) {
        for (int left = 0; left < n - len; ++left) {
            int right = left + len;
            for (int pick = left + 1; pick < right; ++pick) {
                int score = balloons[left] * balloons[pick] * balloons[right] + dp[left][pick] + dp[pick][right];
                if (score > dp[left][right]) {
                    dp[left][right] = score;
                    picks[left][right] = pick;
                }
            }
        }
    }

    // 遞迴重建順序
    vector<int> order;
    MyDP::preorder_312(picks, 0, n - 1, order); //重建：從最後戳破->最開始戳破，所以要reverse
    for (int& idx : order) 
        idx -= 1;
    reverse(order.begin(), order.end());

    vector<int> order_num;
    for (int idx : order) {
        order_num.push_back(nums[idx]);//nums = [3,1,5,8] -> 戳破順序：[1,5,3,8]
    }
    return order_num;
}

void MyDP::preorder_312(const vector<vector<int>>& pick, int left, int right, vector<int>& order) {
    int k = pick[left][right];
    if (k == -1) return;
    order.push_back(k);  // pick 是這段區間最後戳破的氣球
    preorder_312(pick, left, k, order);
    preorder_312(pick, k, right, order);
}

vector<int> MyDP::Top_dowm_Memoization_312_bonus(vector<int>& nums) {
    return {};
}
#pragma endregion

#pragma region TSP (Traveling Salesman Problem) 旅行推銷員問題
/*有一個推銷員要拜訪 n 個城市，每個城市只能拜訪一次，最後回到起點。
  每兩個城市間都有一個距離（或花費）。
  問：如何安排順序，讓總路程最短？*/

/*變形TSP：
一、題目要求：給定一組字串 words，找到一個最短的超字串（superstring），其中包含了所有 words 中的字串。
二、題目限制：每個字串只出現一次。可以讓字串重疊以縮短 superstring 長度。
三、想法：DP + Bitmask (TSP)。
四、作法：如上
五、複雜度：(僅算法使用過程，最壞程度(不包含return ans))
時間複雜度： O(n² × 2ⁿ)
2ⁿ：mask << 1
空間複雜度： O(2 * 2ⁿ * n)
六、難度：*****      */

string MyDP::Leetcode_Sol_943(vector<string>& words, int _solution) {
    switch (_solution)
    {
    case 1:
        return DP_BitMask_TSP_943(words);
    default:
        return ""; // 確保所有路徑都有回傳值
    }

    return "";
}

string MyDP::DP_BitMask_TSP_943(vector<string>& words) {
    int n = words.size();
    // Step 1: 計算重疊矩陣 overlap[i][j] O(n^3)
    //overlap[i][j]：words[j]對於 words[i] 多了多少長度 => ex：{ab,bc}：bc相對於av多了c，也就是多了一個長度
    vector<vector<int>> overlap(n, vector<int>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            int m = min(words[i].size(), words[j].size());
            for (int k = m; k >= 0; --k) {
                if (words[i].substr(words[i].size() - k) == words[j].substr(0, k)) {
                    overlap[i][j] = words[j].size() - k;
                    break;
                }
            }
        }
    }

    // Step 2: DP + Bitmask (mask：代表選了那些word，舉例：0011 = 選了 words[0] 跟 words[1])
    int size = 1 << n, allpick = size - 1;
    //dp[mask][i]：這些組合中，以words[i]為最後一個字串的 Superstring 長度
    vector<vector<int>> dp(size, vector<int>(n, INT_MAX));
    //parent[mask][i]：這些組合中，準備加入words[i]的前一個idx，ex：{ab,b} => 使用b之前的"前一個idx" = "ab的idx"
    vector<vector<int>> parent(size, vector<int>(n, -1));
    //初始化：只選擇自己的word，設定他自己的長度
    for (int i = 0; i < n; ++i)
        dp[1 << i][i] = words[i].size();

    //所有組合遍歷一次 O(n^2 * 2^n)
    for (int mask = 1; mask < size; ++mask) {
        //遍歷一次words，為了只保留當前words[i]，排出 "與當前words組合的前一個組合" => prev_mask (前一個組合)
        for (int i = 0; i < n; ++i) {
            if (!(mask & (1 << i))) continue;
            int prev_mask = mask ^ (1 << i);    //排出當前以外的組合 (因為等等要把它加到最底部)
            if (prev_mask == 0) continue;       //自己排除自己只有自己的組合(那就是沒有使用任何words)
            //遍歷一次words，找尋前一個idx! (看哪一個idx 配合這次的words[i]，字串更短)
            for (int j = 0; j < n; ++j) {
                if (!(prev_mask & (1 << j))) continue;
                int val = dp[prev_mask][j] + overlap[j][i];
                //紀錄這次加入的words[i]的最短字串 與 和他組合的前一個idx
                if (val < dp[mask][i]) {
                    dp[mask][i] = val;
                    parent[mask][i] = j;
                }
            }
        }
    }

    // Step 3: 找最短的結果 (最短 Superstring 長度)
    int min_len = INT_MAX, last = -1;
    for (int i = 0; i < n; ++i) {
        if (dp[allpick][i] < min_len) { //size - 1：全部都選
            min_len = dp[allpick][i];
            last = i;
        }
    }

    // Step 4: 找到組合順序
    vector<int> path;  //紀錄index順序(最後一個為 Superstring 的第一個先寫的字串)
    int mask = allpick;
    while (last != -1) {    //小技巧：last != -1
        path.push_back(last);
        int temp = parent[mask][last];
        mask ^= (1 << last);
        last = temp;
    }

    // Step 5: 寫出 Superstring
    string res = words[path.back()];
    for (int i = path.size() - 1; i > 0; i--) { //i > 0：是因為 path[i - 1]的緣故!
        int overlen = overlap[path[i]][path[i - 1]];
        res += words[path[i - 1]].substr(words[path[i - 1]].size() - overlen);
    }
    return res;
}

#pragma endregion

#pragma region Numbers With Repeated Digits
int MyDP::Leetcode_Sol_1012(int n) {
    int x = n + 1;
    vector<int> digit;
    while (x > 0) {
        digit.emplace_back(x % 10);
        x /= 10;
    }

    //First：Handling less than the maximum number of digits (處理小於最高位數)
    //1000：0 ~ 999 => C(9,1)(最高位：1~9) * C(9,1)(刺位：0~9 - 最高位已選數字(1個)) * C(8,1)(刺位：0~9 - 前面已選數字(2個))
    //=> C(9,1) * P(9,2)
    int res_noduplicate = 0; 
    int lesslen = digit.size();
    int curr_noduplicate = 1;   //當前高位的前一位不重複組合數
    int fac_highestnum = Combination(9,1); //C(9,1) (最高位不能有 0，所以 1~9)
    for (int i = 0; i < lesslen - 1; i++) {
        curr_noduplicate = fac_highestnum * MyDP::Premutation_pick_k(9, i); //9 * 9 * 8 
        res_noduplicate += curr_noduplicate;
    }
    //最高位固定：C(1,1) * P(9,lesslen - 1)
    curr_noduplicate = MyDP::Premutation_pick_k(9, lesslen - 1);
    //Handling the higest digit, start from the highest digit (處理最高位數，從最高位開始)
    vector<int> used(10, false);  //當前高位的數值是否是用過!
    for (int i = 0; i < lesslen; i++) {
        int d = digit[lesslen - 1 - i];
        //Processing the highest digit interval(處理最高位數的區間組合數) => ex：3345：1000~2000、2000~3000 (不會處理 3000 ~ 3345，要到次位數才會處理) 
        //次位數會以上一個高位數的最大值去定型排列 =>ex：3345：次高位數 345 是以 3XXX在處理的 (因為次高位數在處理的是3000~3345)
        for (int j = i == 0 ? 1 : 0; j < d; j++) {
            //因為前面處理小於最高位的時候，已經是次高位的全部組合，所以可以直接加 curr_noduplicate! (curr_noduplicate = P(9,digit.size() - 1 - i))
            if (!used[j])
                res_noduplicate += curr_noduplicate; //C(1,1) * P(9,digit.size() - 1 - i)  => 2000：C(1,1) * P(9,3)
        }

        curr_noduplicate /= fac_highestnum - i;//fac_highestnum - i => 代表最高位已經固定選了一個數，i：在第幾高位，所以" - i"就是減去用了幾個數
        //因為現在在處理最高位數，所以最高位數已經使用的數字就不能使用 => 3356：3000~3356 (使用過 3)-> 356：300~356 (使用過 4)...
        //ex：3345 處理 1000 ~ 2999 (使用過 3) -> 345 處理 3000的 0 ~ 299 (重複 3) -> 45 處理3300的 0 ~ 39 (但這時候，是以3300定型去算的，所以已經重複了!)
        if (used[d]) 
            break; 

        used[d] = true;
    }

    return n - res_noduplicate;
}
/*流程，以3456為例：
=> 3456 處理 未定型的 1000 ~ 2999   (使用過 3) 
=>  456 處理 以 3000 定型的 0 ~ 399 (使用過 4) 
=>   56 處理 以 3400 定型的 0 ~ 49  (使用過 5)
=>    6 處理 以 3450 定型的 0 ~ 6   (使用過 6)*/
#pragma endregion




#pragma endregion

#pragma region Leetcode xxx. ExamName
//Leetcode xxx. ExamName
vector<int> MyDP::Leetcode_Sol_xxx(vector<int>& numbers, int target,int _solution) {
    switch (_solution)
    {
    case 1:
        return Exam_xxx(numbers, target);
    default:
        return std::vector<int>{}; // 確保所有路徑都有回傳值
    }

    return{};
}

vector<int> MyDP::Exam_xxx(vector<int>& numbers, int target) {
    return {};
}
#pragma endregion


