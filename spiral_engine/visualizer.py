"""
visualizer.py - 평가원 수능 스타일 고밀도 가계도 시각화 엔진 (Final Remaster)
- '평가원 폰트/선 굵기/패턴 밀도' 완벽 재현 버전
"""

import io
import base64
import platform
import math
import numpy as np
from typing import Optional
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from .models import Family, Person, Gender


# ============================================================
# 한글 폰트 설정 (시스템 자동 감지)
# ============================================================
def setup_korean_font():
    system = platform.system()
    if system == 'Windows':
        font_name = 'Malgun Gothic'
    elif system == 'Darwin':
        font_name = 'AppleGothic'
    else:
        font_name = 'NanumGothic'  # 리눅스/코랩 등

    plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False
    return font_name


KOREAN_FONT = setup_korean_font()


# ============================================================
# 설정값: 평가원 시험지 비율 (High Density)
# ============================================================
@dataclass
class GridConfig:
    # 캔버스
    fig_width: float = 12.0
    fig_height: float = 7.0

    # Y좌표 (세대 간격은 유지하되 위아래 여백 최소화)
    y_gen1: float = 3.5
    y_gen2: float = 2.0
    y_gen3: float = 0.5

    # [핵심 변경] 간격 압축 (단단한 느낌)
    couple_gap: float = 1.0     # 부부 사이
    sibling_gap: float = 1.0    # 형제 사이
    family_gap: float = 2.0     # 두 집안 사이 (기존보다 훨씬 좁게)

    # 도형 크기
    node_size: float = 0.35     # 도형을 조금 더 키워서 꽉 차게

    # 스타일
    line_width: float = 1.5           # 외곽선 굵기
    pattern_line_width: float = 0.8   # 빗금 굵기 (얇고 촘촘하게)
    edge_color: str = 'black'

    # 색상 팔레트
    color_normal: str = 'white'
    color_gray: str = '#D3D3D3'  # (가)+(나) 동시 발현 시 채울 회색

    font_size_label: int = 12
    font_size_legend: int = 10


# ============================================================
# 시각화 엔진 메인
# ============================================================
class PedigreeVisualizer:
    def __init__(self, config: Optional[GridConfig] = None):
        self.config = config or GridConfig()

    def draw(self, family: Family, title: str = "", save_path: Optional[str] = None) -> str:
        cfg = self.config
        fig, ax = plt.subplots(figsize=(cfg.fig_width, cfg.fig_height))

        # 1. 위치 계산 (Layout)
        self._calculate_positions(family)

        # 2. 연결선 그리기 (Lines)
        self._draw_connections(ax, family)

        # 3. 노드 그리기 (Nodes & Patterns) - 여기가 핵심
        self._draw_nodes(ax, family)

        # 4. 텍스트 & 범례
        self._draw_labels(ax, family)
        self._draw_legend(ax, family)

        # 마무리 설정
        ax.set_aspect('equal')
        ax.axis('off')

        # 여백 최적화
        all_x = [p.x_pos for p in family.members.values()]
        margin = 1.0
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin + 3.0)  # 범례 공간
        ax.set_ylim(-0.5, 4.5)

        plt.tight_layout()

        # 파일 저장
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

        # 이미지 반환
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return img_base64

    # --------------------------------------------------------
    # [1] 좌표 계산 로직 (수능형 3세대 구조)
    # --------------------------------------------------------
    def _calculate_positions(self, family: Family):
        cfg = self.config

        # 1세대 (조부모) 분류
        gen1 = family.get_generation(1)
        couples_g1 = []
        visited = set()
        for p in gen1:
            if p.id not in visited and p.spouse_id:
                sp = family.get_member(p.spouse_id)
                if sp:
                    pair = (p, sp) if p.gender == Gender.MALE else (sp, p)
                    couples_g1.append(pair)
                    visited.add(p.id)
                    visited.add(sp.id)

        left_gp = couples_g1[0] if len(couples_g1) > 0 else (None, None)
        right_gp = couples_g1[1] if len(couples_g1) > 1 else (None, None)

        # 2세대 분류
        gen2 = family.get_generation(2)
        left_kids = [p for p in gen2 if left_gp[0] and p.father_id == left_gp[0].id]
        right_kids = [p for p in gen2 if right_gp[0] and p.father_id == right_gp[0].id]

        # 2세대 결혼 커플
        married_pair_g2 = None
        for p in gen2:
            if p.spouse_id:
                sp = family.get_member(p.spouse_id)
                if sp and sp.generation == 2:
                    # 왼쪽 집안과 오른쪽 집안의 결합 찾기
                    if p in left_kids and sp in right_kids:
                        married_pair_g2 = (p, sp)
                    elif p in right_kids and sp in left_kids:
                        married_pair_g2 = (sp, p)
                    if married_pair_g2:
                        break

        # 중심점 기준 배치
        cx = 6.0
        left_center = cx - cfg.family_gap
        right_center = cx + cfg.family_gap

        # 1. 조부모 배치
        if left_gp[0]:
            left_gp[0].x_pos = left_center - cfg.couple_gap / 2
            left_gp[0].y_pos = cfg.y_gen1
            left_gp[1].x_pos = left_center + cfg.couple_gap / 2
            left_gp[1].y_pos = cfg.y_gen1

        if right_gp[0]:
            right_gp[0].x_pos = right_center - cfg.couple_gap / 2
            right_gp[0].y_pos = cfg.y_gen1
            right_gp[1].x_pos = right_center + cfg.couple_gap / 2
            right_gp[1].y_pos = cfg.y_gen1

        # 2. 2세대 자녀 배치 (결혼한 자녀 제외하고 일단 배치)
        def place_siblings(kids, center):
            if not kids:
                return
            width = (len(kids) - 1) * cfg.sibling_gap
            start = center - width / 2
            for i, k in enumerate(kids):
                k.x_pos = start + i * cfg.sibling_gap
                k.y_pos = cfg.y_gen2

        place_siblings(left_kids, left_center)
        place_siblings(right_kids, right_center)

        # 3. 결혼한 2세대 중앙 정렬 (핵심)
        if married_pair_g2:
            p_left, p_right = married_pair_g2
            # 두 사람을 중앙으로 강제 이동
            p_left.x_pos = cx - cfg.couple_gap / 2
            p_right.x_pos = cx + cfg.couple_gap / 2
            p_left.y_pos = cfg.y_gen2
            p_right.y_pos = cfg.y_gen2

        # 4. 3세대 손자녀 배치
        gen3 = family.get_generation(3)
        if gen3:
            width = (len(gen3) - 1) * cfg.sibling_gap
            start = cx - width / 2
            for i, k in enumerate(gen3):
                k.x_pos = start + i * cfg.sibling_gap
                k.y_pos = cfg.y_gen3

    # --------------------------------------------------------
    # [2] 연결선 그리기 (직각 배선)
    # --------------------------------------------------------
    def _draw_connections(self, ax, family):
        cfg = self.config
        drawn = set()

        for p1 in family.members.values():
            if not p1.spouse_id:
                continue
            p2 = family.get_member(p1.spouse_id)
            if not p2:
                continue

            key = tuple(sorted([p1.id, p2.id]))
            if key in drawn:
                continue
            drawn.add(key)

            # 부부 연결선
            ax.plot([p1.x_pos, p2.x_pos], [p1.y_pos, p2.y_pos],
                    color='black', lw=cfg.line_width, zorder=1)

            # 자녀 연결
            kids = [family.get_member(cid) for cid in p1.children_ids if family.get_member(cid)]
            if kids:
                mx = (p1.x_pos + p2.x_pos) / 2
                my = p1.y_pos
                drop_y = (my + kids[0].y_pos) / 2 + 0.25

                # 수직 내림
                ax.plot([mx, mx], [my, drop_y], color='black', lw=cfg.line_width, zorder=1)

                # 자녀들 수평선 (2명 이상일 때만)
                k_xs = [k.x_pos for k in kids]
                if len(kids) >= 2:
                    ax.plot([min(k_xs), max(k_xs)], [drop_y, drop_y],
                            color='black', lw=cfg.line_width, zorder=1)

                # 자녀들 머리 위 수직선
                for k in kids:
                    # 자녀가 1명이면 부모 중앙에서 바로 수직선
                    if len(kids) == 1:
                        ax.plot([mx, k.x_pos], [drop_y, drop_y],
                                color='black', lw=cfg.line_width, zorder=1)
                    ax.plot([k.x_pos, k.x_pos], [drop_y, k.y_pos + cfg.node_size],
                            color='black', lw=cfg.line_width, zorder=1)

    # --------------------------------------------------------
    # [3] 노드 및 패턴 그리기 (Visual Core)
    # --------------------------------------------------------
    def _draw_nodes(self, ax, family):
        cfg = self.config
        genes = family.genes

        for p in family.members.values():
            x, y = p.x_pos, p.y_pos
            sz = cfg.node_size

            # 물음표(?) 처리
            if p.phenotype_hidden:
                self._draw_shape_base(ax, x, y, sz, p.gender, 'white')
                continue

            # 형질 판단
            has_a = False  # 빗금
            has_b = False  # 체크

            if len(genes) >= 1:
                t = p.get_trait(genes[0].symbol)
                if t and t.is_affected():
                    has_a = True

            if len(genes) >= 2:
                t = p.get_trait(genes[1].symbol)
                if t and t.is_affected():
                    has_b = True

            # === 스타일 적용 ===
            # 우선순위: 둘다(회색) > 하나만(패턴) > 정상(흰색)

            if has_a and has_b:
                # 둘 다 발현 -> 회색 채우기 (패턴 없음)
                self._draw_shape_base(ax, x, y, sz, p.gender, cfg.color_gray)

            elif has_a:
                # (가)만 발현 -> 흰색 바탕 + 빗금
                self._draw_shape_base(ax, x, y, sz, p.gender, 'white')
                if p.gender == Gender.MALE:
                    self._draw_hatch_rect(ax, x, y, sz)
                else:
                    self._draw_hatch_circle(ax, x, y, sz)

            elif has_b:
                # (나)만 발현 -> 흰색 바탕 + 체크(격자)
                self._draw_shape_base(ax, x, y, sz, p.gender, 'white')
                if p.gender == Gender.MALE:
                    self._draw_grid_rect(ax, x, y, sz)
                else:
                    self._draw_grid_circle(ax, x, y, sz)

            else:
                # 정상
                self._draw_shape_base(ax, x, y, sz, p.gender, 'white')

    def _draw_shape_base(self, ax, x, y, sz, gender, color):
        """기본 도형 그리기 (테두리 + 채우기)"""
        if gender == Gender.MALE:
            patch = Rectangle((x - sz, y - sz), sz * 2, sz * 2,
                              facecolor=color, edgecolor='black', lw=1.5, zorder=10)
        else:
            patch = Circle((x, y), sz,
                           facecolor=color, edgecolor='black', lw=1.5, zorder=10)
        ax.add_patch(patch)

    # --- 수제 패턴 함수들 (Matplotlib Hatch 안씀, Clip Path 방식) ---

    def _draw_hatch_rect(self, ax, x, y, sz):
        """사각형 빗금 (Dense) - Clip Path 방식"""
        step = 0.12  # 빗금 간격 (작을수록 촘촘)
        rect = Rectangle((x - sz, y - sz), sz * 2, sz * 2, transform=ax.transData)
        for i in np.arange(-sz * 3, sz * 3, step):
            line, = ax.plot([x - sz * 2, x + sz * 2], [y + i - sz * 2, y + i + sz * 2],
                            color='black', lw=0.8, zorder=11)
            line.set_clip_path(rect)

    def _draw_hatch_circle(self, ax, x, y, sz):
        """원 빗금 (Dense) - Clip Path 방식"""
        step = 0.12
        circ = Circle((x, y), sz, transform=ax.transData)
        for i in np.arange(-sz * 3, sz * 3, step):
            line, = ax.plot([x - sz * 2, x + sz * 2], [y + i - sz * 2, y + i + sz * 2],
                            color='black', lw=0.8, zorder=11)
            line.set_clip_path(circ)

    def _draw_grid_rect(self, ax, x, y, sz):
        """사각형 체크무늬 (Dense) - Clip Path 방식"""
        step = 0.12
        rect = Rectangle((x - sz, y - sz), sz * 2, sz * 2, transform=ax.transData)
        # 수직선
        for i in np.arange(x - sz + step, x + sz, step):
            line, = ax.plot([i, i], [y - sz, y + sz], color='black', lw=0.8, zorder=11)
            line.set_clip_path(rect)
        # 수평선
        for i in np.arange(y - sz + step, y + sz, step):
            line, = ax.plot([x - sz, x + sz], [i, i], color='black', lw=0.8, zorder=11)
            line.set_clip_path(rect)

    def _draw_grid_circle(self, ax, x, y, sz):
        """원 체크무늬 (Dense) - Clip Path 방식"""
        step = 0.12
        circ = Circle((x, y), sz, transform=ax.transData)
        # 수직선
        for i in np.arange(x - sz, x + sz, step):
            line, = ax.plot([i, i], [y - sz, y + sz], color='black', lw=0.8, zorder=11)
            line.set_clip_path(circ)
        # 수평선
        for i in np.arange(y - sz, y + sz, step):
            line, = ax.plot([x - sz, x + sz], [i, i], color='black', lw=0.8, zorder=11)
            line.set_clip_path(circ)

    # --------------------------------------------------------
    # [4] 라벨 및 범례
    # --------------------------------------------------------
    def _draw_labels(self, ax, family):
        cfg = self.config
        for p in family.members.values():
            if p.display_name:
                ax.text(p.x_pos, p.y_pos - cfg.node_size - 0.2, p.display_name,
                        ha='center', va='top',
                        fontsize=cfg.font_size_label, fontweight='bold',
                        fontfamily=KOREAN_FONT)

    def _draw_legend(self, ax, family):
        cfg = self.config
        # 범례 위치 (오른쪽 구석)
        lx = max([p.x_pos for p in family.members.values()]) + 1.5
        ly = cfg.y_gen1

        genes = family.genes
        n1 = genes[0].trait_name if len(genes) > 0 else "(가)"
        n2 = genes[1].trait_name if len(genes) > 1 else "(나)"

        items = [
            ("정상 남자", 'white', False, False, Gender.MALE),
            ("정상 여자", 'white', False, False, Gender.FEMALE),
            (f"{n1} 발현 남자", 'white', True, False, Gender.MALE),
            (f"{n1} 발현 여자", 'white', True, False, Gender.FEMALE),
            (f"{n2} 발현 남자", 'white', False, True, Gender.MALE),
            (f"{n2} 발현 여자", 'white', False, True, Gender.FEMALE),
        ]

        sz = 0.15  # 범례 아이콘 크기
        for i, (txt, col, h_a, h_b, gen) in enumerate(items):
            cy = ly - i * 0.5

            # 아이콘 그리기
            self._draw_shape_base(ax, lx, cy, sz, gen, col)
            if h_a:
                if gen == Gender.MALE:
                    self._draw_hatch_rect(ax, lx, cy, sz)
                else:
                    self._draw_hatch_circle(ax, lx, cy, sz)
            if h_b:
                if gen == Gender.MALE:
                    self._draw_grid_rect(ax, lx, cy, sz)
                else:
                    self._draw_grid_circle(ax, lx, cy, sz)

            ax.text(lx + 0.4, cy, txt, va='center', fontsize=cfg.font_size_legend,
                    fontfamily=KOREAN_FONT)

    # --------------------------------------------------------
    # 유틸리티 메서드
    # --------------------------------------------------------
    def save_to_file(self, family: Family, filepath: str, title: str = ""):
        """파일로 저장"""
        self.draw(family, title=title, save_path=filepath)

    def get_base64_image(self, family: Family, title: str = "") -> str:
        """Base64 이미지 반환"""
        return self.draw(family, title=title)
