def solve():
    import sys
    sys.setrecursionlimit(10000)

    # ---------- Geometry Helpers ----------
    def orientation(p, q, r):
        """Return orientation of ordered triplet (p, q, r):
        0 = collinear, 1 = clockwise, 2 = counterclockwise"""
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-9:
            return 0
        return 1 if val > 0 else 2

    def on_segment(p, q, r):
        """Check if q lies on segment pr"""
        return (min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9 and
                min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9)

    def intersect(p1, q1, p2, q2):
        """Return intersection point if segments intersect, else None"""
        o1, o2 = orientation(p1, q1, p2), orientation(p1, q1, q2)
        o3, o4 = orientation(p2, q2, p1), orientation(p2, q2, q1)

        # General case
        if o1 != o2 and o3 != o4:
            x1, y1, x2, y2 = *p1, *q1
            x3, y3, x4, y4 = *p2, *q2
            A1, B1, C1 = y2 - y1, x1 - x2, (y2 - y1) * x1 + (x1 - x2) * y1
            A2, B2, C2 = y4 - y3, x3 - x4, (y4 - y3) * x3 + (x3 - x4) * y3
            det = A1 * B2 - A2 * B1
            if abs(det) < 1e-9:
                return None
            x = (B2 * C1 - B1 * C2) / det
            y = (A1 * C2 - A2 * C1) / det
            return (round(x, 6), round(y, 6))

        # Special cases: collinear
        if o1 == 0 and on_segment(p1, p2, q1): return p2
        if o2 == 0 and on_segment(p1, q2, q1): return q2
        if o3 == 0 and on_segment(p2, p1, q2): return p1
        if o4 == 0 and on_segment(p2, q1, q2): return q1
        return None

    # ---------- Input ----------
    n = int(sys.stdin.readline().strip())
    segments = []
    for _ in range(n):
        x1, y1, x2, y2 = map(int, sys.stdin.readline().split())
        segments.append(((x1, y1), (x2, y2)))

    # Step 1: find intersections and store them
    seg_points = {i: [segments[i][0], segments[i][1]] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            p1, q1 = segments[i]
            p2, q2 = segments[j]
            inter = intersect(p1, q1, p2, q2)
            if inter:
                if inter not in seg_points[i]:
                    seg_points[i].append(inter)
                if inter not in seg_points[j]:
                    seg_points[j].append(inter)

    # Step 2: split into small edges
    edges = set()
    points = set()
    for i in range(n):
        pts = seg_points[i]
        p1, p2 = segments[i]
        if abs(p1[0] - p2[0]) < 1e-9:  # vertical
            pts.sort(key=lambda p: p[1])
        else:
            pts.sort(key=lambda p: p[0])
        for k in range(len(pts) - 1):
            u, v = pts[k], pts[k + 1]
            if u != v:
                edges.add((u, v))
                edges.add((v, u))
                points.add(u)
                points.add(v)

    # Step 3: build graph
    graph = {p: [] for p in points}
    for u, v in edges:
        graph[u].append(v)

    # Step 4: count connected components
    visited = set()
    def dfs(node):
        stack = [node]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            for nei in graph[u]:
                if nei not in visited:
                    stack.append(nei)

    C = 0
    for p in points:
        if p not in visited:
            C += 1
            dfs(p)

    V = len(points)
    E = len(edges) // 2  # each edge stored twice
    closed_shapes = E - V + C
    print(closed_shapes)


if __name__ == "__main__":
    solve()
