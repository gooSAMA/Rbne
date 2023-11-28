"""
    使用LeaderRank算法
"""

from leaderrank import LeaderRank

def run(matrix):
    return LeaderRank(matrix).leaderRank()
