"""
visualizer.py - 가계도 시각화
networkx와 matplotlib를 사용한 가계도 그리기
"""

import io
import base64
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np

from .models import Family, Person, Gender, Phenotype, Gene


@dataclass
class VisualizationConfig:
    """시각화 설정"""
    # 캔버스 크기
    fig_width: float = 14
    fig_height: float = 10

    # 노드 크기
    node_size: float = 0.4

    # 색상
    color_normal: str = '#FFFFFF'        # 정상 (흰색)
    color_affected: str = '#808080'      # 발현 (회색)
    color_carrier: str = '#CCCCCC'       # 보인자 표시용
    color_edge: str = '#000000'          # 테두리
    color_line: str = '#000000'          # 연결선
    color_marriage: str = '#000000'      # 결혼선

    # 텍스트
    font_family: str = 'DejaVu Sans'
    font_size_label: int = 10
    font_size_genotype: int = 8

    # 레이아웃
    generation_gap: float = 2.5          # 세대 간 간격
    sibling_gap: float = 1.8             # 형제 간 간격
    couple_gap: float = 1.2              # 부부 간 간격

    # 스타일
    show_genotype: bool = True           # 유전자형 표시
    show_carrier_pattern: bool = True    # 보인자 패턴 표시
    show_legend: bool = True             # 범례 표시


class PedigreeVisualizer:
    """
    가계도 시각화 클래스
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        plt.rcParams['font.family'] = self.config.font_family

    def draw_pedigree(
        self,
        family: Family,
        gene_symbol: Optional[str] = None,
        title: str = "가계도",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[str]:
        """
        가계도 그리기

        Args:
            family: 가족 객체
            gene_symbol: 표시할 유전자 기호 (None이면 첫 번째 형질)
            title: 제목
            save_path: 저장 경로
            show: 화면 표시 여부

        Returns:
            base64 인코딩된 이미지 (웹 서비스용)
        """
        fig, ax = plt.subplots(
            figsize=(self.config.fig_width, self.config.fig_height)
        )

        # 위치 계산
        positions = self._calculate_positions(family)

        # 연결선 그리기 (노드 뒤에)
        self._draw_connections(ax, family, positions)

        # 노드 그리기
        self._draw_nodes(ax, family, positions, gene_symbol)

        # 범례
        if self.config.show_legend:
            self._draw_legend(ax, gene_symbol)

        # 제목
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # 축 설정
        ax.set_aspect('equal')
        ax.axis('off')

        # 여백 조정
        plt.tight_layout()

        # 저장
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')

        # base64 인코딩 (웹용)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        if show:
            plt.show()
        else:
            plt.close()

        return img_base64

    def _calculate_positions(self, family: Family) -> Dict[str, Tuple[float, float]]:
        """각 구성원의 위치 계산"""
        positions = {}
        cfg = self.config

        # 세대별 Y 좌표
        y_positions = {
            0: cfg.generation_gap * 2,   # 조부모
            1: cfg.generation_gap,       # 부모
            2: 0                          # 자녀
        }

        # 0세대: 조부모 (양가)
        gen0 = family.get_generation(0)
        # 친가 조부모 (왼쪽)
        paternal_grandparents = []
        maternal_grandparents = []

        for p in gen0:
            # 부모 세대 확인하여 친가/외가 구분
            for child in family.get_children(p.id):
                if child.gender == Gender.MALE and child.generation == 1:
                    paternal_grandparents.append(p)
                elif child.gender == Gender.FEMALE and child.generation == 1:
                    maternal_grandparents.append(p)

        # 친가 조부모 위치 (왼쪽)
        x_paternal_center = -cfg.couple_gap * 1.5
        for i, p in enumerate(paternal_grandparents):
            offset = -cfg.couple_gap / 2 if i == 0 else cfg.couple_gap / 2
            positions[p.id] = (x_paternal_center + offset, y_positions[0])

        # 외가 조부모 위치 (오른쪽)
        x_maternal_center = cfg.couple_gap * 1.5
        for i, p in enumerate(maternal_grandparents):
            offset = -cfg.couple_gap / 2 if i == 0 else cfg.couple_gap / 2
            positions[p.id] = (x_maternal_center + offset, y_positions[0])

        # 1세대: 부모
        gen1 = family.get_generation(1)
        father = next((p for p in gen1 if p.gender == Gender.MALE), None)
        mother = next((p for p in gen1 if p.gender == Gender.FEMALE), None)

        if father:
            positions[father.id] = (-cfg.couple_gap / 2, y_positions[1])
        if mother:
            positions[mother.id] = (cfg.couple_gap / 2, y_positions[1])

        # 2세대: 자녀
        gen2 = family.get_generation(2)
        num_children = len(gen2)

        if num_children > 0:
            # 자녀들을 중앙 정렬
            total_width = (num_children - 1) * cfg.sibling_gap
            start_x = -total_width / 2

            for i, child in enumerate(gen2):
                x = start_x + i * cfg.sibling_gap
                positions[child.id] = (x, y_positions[2])

        return positions

    def _draw_nodes(
        self,
        ax: plt.Axes,
        family: Family,
        positions: Dict[str, Tuple[float, float]],
        gene_symbol: Optional[str] = None
    ):
        """구성원 노드 그리기"""
        cfg = self.config

        for member in family.all_members:
            if member.id not in positions:
                continue

            x, y = positions[member.id]

            # 표현형 확인
            phenotype = Phenotype.DOMINANT
            is_carrier = False

            if gene_symbol:
                trait = member.get_trait(gene_symbol)
                if trait:
                    phenotype = trait.phenotype
                    is_carrier = trait.is_carrier
            elif member.traits:
                # 첫 번째 형질 사용
                first_trait = list(member.traits.values())[0]
                phenotype = first_trait.phenotype
                is_carrier = first_trait.is_carrier

            # 색상 결정
            if phenotype == Phenotype.RECESSIVE:
                fill_color = cfg.color_affected
            else:
                fill_color = cfg.color_normal

            # 노드 그리기
            if member.gender == Gender.MALE:
                # 남성: 사각형
                self._draw_square(ax, x, y, cfg.node_size, fill_color,
                                 is_carrier and cfg.show_carrier_pattern)
            else:
                # 여성: 원
                self._draw_circle(ax, x, y, cfg.node_size, fill_color,
                                 is_carrier and cfg.show_carrier_pattern)

            # 라벨 (이름/ID)
            label = member.display_name or member.id
            ax.text(x, y - cfg.node_size - 0.3, label,
                   ha='center', va='top',
                   fontsize=cfg.font_size_label)

            # 유전자형 표시
            if cfg.show_genotype and gene_symbol:
                genotype = member.get_genotype(gene_symbol)
                if genotype:
                    ax.text(x, y + cfg.node_size + 0.2, genotype,
                           ha='center', va='bottom',
                           fontsize=cfg.font_size_genotype,
                           style='italic', color='#666666')

    def _draw_square(
        self,
        ax: plt.Axes,
        x: float, y: float,
        size: float,
        fill_color: str,
        show_carrier_pattern: bool = False
    ):
        """사각형 (남성) 그리기"""
        cfg = self.config
        half = size / 2

        rect = Rectangle(
            (x - half, y - half), size, size,
            facecolor=fill_color,
            edgecolor=cfg.color_edge,
            linewidth=2
        )
        ax.add_patch(rect)

        # 보인자 패턴 (대각선)
        if show_carrier_pattern:
            ax.plot([x - half, x + half], [y - half, y + half],
                   color=cfg.color_edge, linewidth=1.5)

    def _draw_circle(
        self,
        ax: plt.Axes,
        x: float, y: float,
        size: float,
        fill_color: str,
        show_carrier_pattern: bool = False
    ):
        """원 (여성) 그리기"""
        cfg = self.config
        radius = size / 2

        circle = Circle(
            (x, y), radius,
            facecolor=fill_color,
            edgecolor=cfg.color_edge,
            linewidth=2
        )
        ax.add_patch(circle)

        # 보인자 패턴 (반원 음영 또는 점)
        if show_carrier_pattern:
            # 점으로 표시
            ax.plot(x, y, 'o', color=cfg.color_edge, markersize=4)

    def _draw_connections(
        self,
        ax: plt.Axes,
        family: Family,
        positions: Dict[str, Tuple[float, float]]
    ):
        """연결선 그리기 (결혼선, 부모-자녀 선)"""
        cfg = self.config

        # 결혼선
        for couple in family.couples:
            p1_id, p2_id = couple
            if p1_id in positions and p2_id in positions:
                x1, y1 = positions[p1_id]
                x2, y2 = positions[p2_id]

                # 수평선
                ax.plot([x1, x2], [y1, y2],
                       color=cfg.color_marriage, linewidth=2)

        # 부모-자녀 연결
        gen1 = family.get_generation(1)
        father = next((p for p in gen1 if p.gender == Gender.MALE), None)
        mother = next((p for p in gen1 if p.gender == Gender.FEMALE), None)

        if father and mother and father.id in positions and mother.id in positions:
            fx, fy = positions[father.id]
            mx, my = positions[mother.id]

            # 부모 중간점
            parent_mid_x = (fx + mx) / 2
            parent_mid_y = fy  # 같은 Y 좌표

            children = family.get_children(father.id)
            if children:
                # 부모에서 아래로 수직선
                child_y = positions[children[0].id][1] if children[0].id in positions else 0
                mid_y = (parent_mid_y + child_y) / 2

                ax.plot([parent_mid_x, parent_mid_x], [parent_mid_y, mid_y],
                       color=cfg.color_line, linewidth=1.5)

                # 자녀들의 X 범위
                child_positions = [positions[c.id] for c in children if c.id in positions]
                if child_positions:
                    min_x = min(cp[0] for cp in child_positions)
                    max_x = max(cp[0] for cp in child_positions)

                    # 수평선 (자녀들 연결)
                    ax.plot([min_x, max_x], [mid_y, mid_y],
                           color=cfg.color_line, linewidth=1.5)

                    # 각 자녀로 수직선
                    for child in children:
                        if child.id in positions:
                            cx, cy = positions[child.id]
                            ax.plot([cx, cx], [mid_y, cy + cfg.node_size / 2],
                                   color=cfg.color_line, linewidth=1.5)

        # 조부모-부모 연결 (양쪽)
        self._draw_grandparent_connections(ax, family, positions, father, 'paternal')
        self._draw_grandparent_connections(ax, family, positions, mother, 'maternal')

    def _draw_grandparent_connections(
        self,
        ax: plt.Axes,
        family: Family,
        positions: Dict[str, Tuple[float, float]],
        parent: Optional[Person],
        side: str
    ):
        """조부모-부모 연결선"""
        if not parent or parent.id not in positions:
            return

        cfg = self.config
        gf, gm = family.get_parents(parent.id)

        if gf and gm and gf.id in positions and gm.id in positions:
            gfx, gfy = positions[gf.id]
            gmx, gmy = positions[gm.id]
            px, py = positions[parent.id]

            # 조부모 중간점
            gp_mid_x = (gfx + gmx) / 2
            mid_y = (gfy + py) / 2

            # 조부모에서 아래로
            ax.plot([gp_mid_x, gp_mid_x], [gfy - cfg.node_size / 2, mid_y],
                   color=cfg.color_line, linewidth=1.5)

            # 수평으로 부모 X 위치까지
            ax.plot([gp_mid_x, px], [mid_y, mid_y],
                   color=cfg.color_line, linewidth=1.5)

            # 부모로 수직선
            ax.plot([px, px], [mid_y, py + cfg.node_size / 2],
                   color=cfg.color_line, linewidth=1.5)

    def _draw_legend(self, ax: plt.Axes, gene_symbol: Optional[str] = None):
        """범례 그리기"""
        cfg = self.config

        legend_elements = [
            mpatches.Patch(facecolor=cfg.color_normal,
                          edgecolor=cfg.color_edge, label='정상'),
            mpatches.Patch(facecolor=cfg.color_affected,
                          edgecolor=cfg.color_edge, label='발현'),
        ]

        # 성별 범례
        legend_elements.append(
            mpatches.Rectangle((0, 0), 1, 1, facecolor='white',
                               edgecolor=cfg.color_edge, label='□ 남성')
        )
        legend_elements.append(
            mpatches.Circle((0.5, 0.5), 0.5, facecolor='white',
                           edgecolor=cfg.color_edge, label='○ 여성')
        )

        ax.legend(handles=legend_elements, loc='upper right',
                 fontsize=9, framealpha=0.9)

    def create_problem_image(
        self,
        family: Family,
        gene_symbol: Optional[str] = None,
        hide_genotypes: bool = True,
        title: str = "유전 가계도"
    ) -> str:
        """
        문제용 이미지 생성 (유전자형 숨김)

        Returns:
            base64 인코딩된 이미지
        """
        # 설정 임시 변경
        original_show_genotype = self.config.show_genotype
        self.config.show_genotype = not hide_genotypes

        img_base64 = self.draw_pedigree(
            family, gene_symbol, title,
            save_path=None, show=False
        )

        # 설정 복원
        self.config.show_genotype = original_show_genotype

        return img_base64

    def create_answer_image(
        self,
        family: Family,
        gene_symbol: Optional[str] = None,
        title: str = "유전 가계도 (정답)"
    ) -> str:
        """
        정답용 이미지 생성 (유전자형 표시)

        Returns:
            base64 인코딩된 이미지
        """
        # 설정 임시 변경
        original_show_genotype = self.config.show_genotype
        self.config.show_genotype = True

        img_base64 = self.draw_pedigree(
            family, gene_symbol, title,
            save_path=None, show=False
        )

        # 설정 복원
        self.config.show_genotype = original_show_genotype

        return img_base64

    def save_pedigree(
        self,
        family: Family,
        filepath: str,
        gene_symbol: Optional[str] = None,
        title: str = "가계도"
    ):
        """가계도를 파일로 저장"""
        self.draw_pedigree(
            family, gene_symbol, title,
            save_path=filepath, show=False
        )
