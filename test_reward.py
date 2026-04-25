import sys
sys.path.insert(0, "/mnt/sdc/ubuntu/cjz_projects/DRIVE")
from reward_verifier import compute_reward

# 用例 1：完美回答，期望得分 1.1
response1 = (
    "<think>这是一道简单的加法题</think>\n"
    "```python\n"
    "import sys\n"
    "print(sum(map(int, sys.stdin.read().split())))\n"
    "```"
)
test_cases = [{"input": "1 2", "output": "3"}, {"input": "10 20", "output": "30"}]

score1 = compute_reward(response1, test_cases)
print(f"用例1（完美回答）得分: {score1}  期望: 1.1")

# 用例 2：死循环/超时，期望得分 0.1
response2 = (
    "<think>思考中</think>\n"
    "```python\n"
    "while True: pass\n"
    "```"
)
score2 = compute_reward(response2, test_cases)
print(f"用例2（超时代码）得分: {score2}  期望: 0.1")

# 用例 3：无 <think> 标签但代码正确，期望得分 1.0
response3 = (
    "```python\n"
    "import sys\n"
    "print(sum(map(int, sys.stdin.read().split())))\n"
    "```"
)
score3 = compute_reward(response3, test_cases)
print(f"用例3（无格式但代码正确）得分: {score3}  期望: 1.0")

# 用例 4：部分通过（只过1/2），期望得分 0.5+0.1=0.6
partial_cases = [{"input": "1 2", "output": "3"}, {"input": "1 2", "output": "999"}]
score4 = compute_reward(response1, partial_cases)
print(f"用例4（部分通过1/2）得分: {score4}  期望: 0.6")

# 用例 5：test_cases 为空，期望得分 0.1
score5 = compute_reward(response1, [])
print(f"用例5（空test_cases）得分: {score5}  期望: 0.1")
