# Scheme heavily adapted from https://github.com/deepmind/pycolab/
# '@' means "wall"
# 'P' means "player" spawn point
# 'A' means apple spawn point
# '' is empty space

# Cleanup colors
# 'H' is potential waste spawn point
# 'R' is river cell
# 'S' is stream cell


# Agent indices are 0,...,N from top to bottom
# 7x7 map: Agent 0 on river side, Agent 1 on apple side
CLEANUP_SMALL_SYM = [
    '@@@@@@@',
    '@H  PB@',
    '@H   B@',
    '@    B@',
    '@    B@',
    '@ P  B@',
    '@@@@@@@']
