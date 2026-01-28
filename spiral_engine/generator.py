"""
generator.py - 가계도 생성기 (역추적 알고리즘)
PedigreeGenerator 클래스
"""

import random
from typing import List, Dict, Optional, Tuple, Set
from copy import deepcopy
from itertools import product

from .models import (
    Gene, GeneticTrait, Person, Family,
    Gender, Phenotype, ChromosomeType
)
from .genetics import InheritanceRule, InheritanceMode


class GenerationConfig:
    """세대별 생성 설정"""

    def __init__(
        self,
        num_couples_gen0: int = 2,      # 조부모 부부 수 (양가)
        num_children_gen1: int = 2,      # 1세대(부모 세대) 자녀 수
        num_children_gen2: int = 3,      # 2세대(자녀 세대) 자녀 수
        ensure_both_genders_gen2: bool = True  # 자녀 중 남녀 모두 포함
    ):
        self.num_couples_gen0 = num_couples_gen0
        self.num_children_gen1 = num_children_gen1
        self.num_children_gen2 = num_children_gen2
        self.ensure_both_genders_gen2 = ensure_both_genders_gen2


class PedigreeConstraint:
    """가계도 생성 시 제약 조건"""

    def __init__(
        self,
        gene: Gene,
        target_person_id: Optional[str] = None,
        target_generation: Optional[int] = None,
        required_phenotype: Optional[Phenotype] = None,
        required_carrier: Optional[bool] = None,
        min_affected_count: int = 0,
        affected_in_generation: Optional[int] = None
    ):
        self.gene = gene
        self.target_person_id = target_person_id
        self.target_generation = target_generation
        self.required_phenotype = required_phenotype
        self.required_carrier = required_carrier
        self.min_affected_count = min_affected_count
        self.affected_in_generation = affected_in_generation


class PedigreeGenerator:
    """
    가계도 생성기 - 역추적(Backtracking) 알고리즘 사용
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.max_attempts = 1000  # 최대 시도 횟수

    def generate_family_structure(
        self,
        config: Optional[GenerationConfig] = None
    ) -> Family:
        """
        가족 구조만 생성 (유전자형 없이)
        3세대 가계도 구조 생성
        """
        if config is None:
            config = GenerationConfig()

        family = Family()
        person_counter = 1

        # === 0세대: 조부모 (양가) ===
        # 아버지측 조부모
        grandpa_f = Person(
            id=f"P{person_counter}",
            gender=Gender.MALE,
            generation=0,
            display_name="친조부"
        )
        person_counter += 1
        grandma_f = Person(
            id=f"P{person_counter}",
            gender=Gender.FEMALE,
            generation=0,
            display_name="친조모"
        )
        person_counter += 1

        # 어머니측 조부모
        grandpa_m = Person(
            id=f"P{person_counter}",
            gender=Gender.MALE,
            generation=0,
            display_name="외조부"
        )
        person_counter += 1
        grandma_m = Person(
            id=f"P{person_counter}",
            gender=Gender.FEMALE,
            generation=0,
            display_name="외조모"
        )
        person_counter += 1

        for p in [grandpa_f, grandma_f, grandpa_m, grandma_m]:
            family.add_member(p)

        family.add_couple(grandpa_f.id, grandma_f.id)
        family.add_couple(grandpa_m.id, grandma_m.id)

        # === 1세대: 부모 ===
        # 아버지 (친조부모의 자녀)
        father = Person(
            id=f"P{person_counter}",
            gender=Gender.MALE,
            generation=1,
            display_name="아버지"
        )
        person_counter += 1

        # 어머니 (외조부모의 자녀)
        mother = Person(
            id=f"P{person_counter}",
            gender=Gender.FEMALE,
            generation=1,
            display_name="어머니"
        )
        person_counter += 1

        family.add_member(father)
        family.add_member(mother)

        family.add_child(father.id, grandpa_f.id, grandma_f.id)
        family.add_child(mother.id, grandpa_m.id, grandma_m.id)
        family.add_couple(father.id, mother.id)

        # === 2세대: 자녀 ===
        children_genders = self._generate_children_genders(
            config.num_children_gen2,
            config.ensure_both_genders_gen2
        )

        for i, gender in enumerate(children_genders):
            child = Person(
                id=f"P{person_counter}",
                gender=gender,
                generation=2,
                display_name=f"자녀{i+1}"
            )
            person_counter += 1
            family.add_member(child)
            family.add_child(child.id, father.id, mother.id)

        return family

    def _generate_children_genders(
        self,
        num_children: int,
        ensure_both: bool
    ) -> List[Gender]:
        """자녀 성별 생성"""
        if num_children < 2:
            return [random.choice([Gender.MALE, Gender.FEMALE])
                    for _ in range(num_children)]

        if ensure_both:
            # 최소 한 명씩 보장
            genders = [Gender.MALE, Gender.FEMALE]
            for _ in range(num_children - 2):
                genders.append(random.choice([Gender.MALE, Gender.FEMALE]))
            random.shuffle(genders)
            return genders
        else:
            return [random.choice([Gender.MALE, Gender.FEMALE])
                    for _ in range(num_children)]

    def assign_genotypes_with_backtracking(
        self,
        family: Family,
        genes: List[Gene],
        constraints: List[PedigreeConstraint]
    ) -> Tuple[bool, Family]:
        """
        역추적 알고리즘을 사용하여 유전자형 할당

        1. 먼저 제약 조건(예: 특정 자녀가 발병)을 설정
        2. 이 조건을 만족시킬 수 있는 부모 유전자형을 역추적
        3. 조부모까지 확장
        4. 모순 발생 시 재시도

        Returns:
            (success, family_with_genotypes)
        """
        family = family.copy()
        family.genes = genes

        for attempt in range(self.max_attempts):
            success = True
            temp_family = family.copy()

            for gene in genes:
                if not self._assign_gene_with_backtracking(temp_family, gene, constraints):
                    success = False
                    break

            if success and self._validate_all_constraints(temp_family, constraints):
                return True, temp_family

        return False, family

    def _assign_gene_with_backtracking(
        self,
        family: Family,
        gene: Gene,
        constraints: List[PedigreeConstraint]
    ) -> bool:
        """단일 유전자에 대한 유전자형 할당 (역추적)"""

        # 해당 유전자 관련 제약조건 필터링
        gene_constraints = [c for c in constraints if c.gene.symbol == gene.symbol]

        # Step 1: 2세대(자녀) 표현형 결정 (제약조건 기반)
        children = family.get_generation(2)
        child_phenotypes = self._determine_child_phenotypes(
            children, gene, gene_constraints
        )

        # Step 2: 자녀 표현형으로부터 가능한 자녀 유전자형 결정
        child_genotypes = {}
        for child in children:
            phenotype = child_phenotypes.get(child.id, Phenotype.DOMINANT)
            possible = self._get_genotypes_for_phenotype(gene, phenotype, child.gender)
            if not possible:
                return False
            child_genotypes[child.id] = random.choice(possible)

        # Step 3: 자녀 유전자형으로부터 부모 유전자형 역추적
        parents = family.get_generation(1)
        father = next((p for p in parents if p.gender == Gender.MALE), None)
        mother = next((p for p in parents if p.gender == Gender.FEMALE), None)

        if not father or not mother:
            return False

        parent_genotypes = self._backtrack_parent_genotypes(
            gene, children, child_genotypes, father, mother
        )

        if not parent_genotypes:
            return False

        father_genotype, mother_genotype = parent_genotypes

        # Step 4: 부모 유전자형으로부터 조부모 유전자형 역추적
        grandparents_f = family.get_parents(father.id)
        grandparents_m = family.get_parents(mother.id)

        gf_genotype = self._backtrack_single_parent_genotype(
            gene, father_genotype, father.gender, Gender.MALE
        )
        gm_f_genotype = self._backtrack_single_parent_genotype(
            gene, father_genotype, father.gender, Gender.FEMALE
        )
        gf_m_genotype = self._backtrack_single_parent_genotype(
            gene, mother_genotype, mother.gender, Gender.MALE
        )
        gm_m_genotype = self._backtrack_single_parent_genotype(
            gene, mother_genotype, mother.gender, Gender.FEMALE
        )

        # Step 5: 모든 구성원에게 유전자형 할당
        # 조부모 (친가)
        if grandparents_f[0]:  # 친조부
            self._assign_trait(grandparents_f[0], gene, gf_genotype)
        if grandparents_f[1]:  # 친조모
            self._assign_trait(grandparents_f[1], gene, gm_f_genotype)

        # 조부모 (외가)
        if grandparents_m[0]:  # 외조부
            self._assign_trait(grandparents_m[0], gene, gf_m_genotype)
        if grandparents_m[1]:  # 외조모
            self._assign_trait(grandparents_m[1], gene, gm_m_genotype)

        # 부모
        self._assign_trait(father, gene, father_genotype)
        self._assign_trait(mother, gene, mother_genotype)

        # 자녀
        for child in children:
            self._assign_trait(child, gene, child_genotypes[child.id])

        return True

    def _determine_child_phenotypes(
        self,
        children: List[Person],
        gene: Gene,
        constraints: List[PedigreeConstraint]
    ) -> Dict[str, Phenotype]:
        """제약조건에 따라 자녀 표현형 결정"""
        phenotypes = {}

        # 기본값: 랜덤 (대부분 정상)
        for child in children:
            phenotypes[child.id] = Phenotype.DOMINANT

        # 제약조건 적용
        for constraint in constraints:
            if constraint.target_person_id:
                if constraint.required_phenotype:
                    phenotypes[constraint.target_person_id] = constraint.required_phenotype

            if constraint.affected_in_generation == 2:
                # 2세대에서 최소 N명 발병
                affected_needed = constraint.min_affected_count
                available = [c for c in children]
                random.shuffle(available)

                for i, child in enumerate(available):
                    if i < affected_needed:
                        phenotypes[child.id] = Phenotype.RECESSIVE

        return phenotypes

    def _get_genotypes_for_phenotype(
        self,
        gene: Gene,
        phenotype: Phenotype,
        gender: Gender
    ) -> List[Tuple[str, Optional[str]]]:
        """특정 표현형을 나타내는 유전자형 목록"""
        d = gene.dominant_allele
        r = gene.recessive_allele

        if gene.chromosome_type == ChromosomeType.X_LINKED:
            if gender == Gender.MALE:
                # 남성: 반성유전, X 하나만
                if phenotype == Phenotype.RECESSIVE:
                    return [(r, None)]
                else:
                    return [(d, None)]
            else:
                # 여성: X 두 개
                if phenotype == Phenotype.RECESSIVE:
                    return [(r, r)]
                else:
                    return [(d, d), (d, r)]
        else:
            # 상염색체
            if phenotype == Phenotype.RECESSIVE:
                return [(r, r)]
            else:
                return [(d, d), (d, r)]

    def _backtrack_parent_genotypes(
        self,
        gene: Gene,
        children: List[Person],
        child_genotypes: Dict[str, Tuple[str, Optional[str]]],
        father: Person,
        mother: Person
    ) -> Optional[Tuple[Tuple[str, Optional[str]], Tuple[str, Optional[str]]]]:
        """
        자녀 유전자형으로부터 가능한 부모 유전자형 역추적

        모든 자녀의 유전자형을 만들 수 있는 부모 유전자형 조합을 찾음
        """
        d = gene.dominant_allele
        r = gene.recessive_allele

        # 가능한 부모 유전자형 목록
        if gene.chromosome_type == ChromosomeType.X_LINKED:
            father_options = [(d, None), (r, None)]
            mother_options = [(d, d), (d, r), (r, r)]
        else:
            father_options = [(d, d), (d, r), (r, r)]
            mother_options = [(d, d), (d, r), (r, r)]

        # 모든 조합 시도
        valid_combinations = []

        for f_geno in father_options:
            for m_geno in mother_options:
                if self._can_parents_produce_all_children(
                    gene, f_geno, m_geno, children, child_genotypes
                ):
                    valid_combinations.append((f_geno, m_geno))

        if not valid_combinations:
            return None

        return random.choice(valid_combinations)

    def _can_parents_produce_all_children(
        self,
        gene: Gene,
        father_genotype: Tuple[str, Optional[str]],
        mother_genotype: Tuple[str, Optional[str]],
        children: List[Person],
        child_genotypes: Dict[str, Tuple[str, Optional[str]]]
    ) -> bool:
        """부모 유전자형이 모든 자녀 유전자형을 생성할 수 있는지 확인"""

        # 부모에서 가능한 배우자 생성
        if father_genotype[1] is not None:
            father_gametes = list(set([father_genotype[0], father_genotype[1]]))
        else:
            father_gametes = [father_genotype[0]]

        if mother_genotype[1] is not None:
            mother_gametes = list(set([mother_genotype[0], mother_genotype[1]]))
        else:
            mother_gametes = [mother_genotype[0]]

        for child in children:
            child_geno = child_genotypes[child.id]

            if gene.chromosome_type == ChromosomeType.X_LINKED:
                if child.gender == Gender.MALE:
                    # 아들: 어머니 X만 받음
                    if child_geno[0] not in mother_gametes:
                        return False
                else:
                    # 딸: 아버지 X + 어머니 X
                    child_alleles = set([child_geno[0], child_geno[1]])
                    # 아버지에게서 하나, 어머니에게서 하나 받아야 함
                    found = False
                    for f_allele in father_gametes:
                        for m_allele in mother_gametes:
                            if set([f_allele, m_allele]) == child_alleles:
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        return False
            else:
                # 상염색체
                child_alleles = set([child_geno[0], child_geno[1]])
                found = False
                for f_allele in father_gametes:
                    for m_allele in mother_gametes:
                        if set([f_allele, m_allele]) == child_alleles:
                            found = True
                            break
                    if found:
                        break
                if not found:
                    return False

        return True

    def _backtrack_single_parent_genotype(
        self,
        gene: Gene,
        child_genotype: Tuple[str, Optional[str]],
        child_gender: Gender,
        parent_gender: Gender
    ) -> Tuple[str, Optional[str]]:
        """단일 자녀의 유전자형으로부터 부모 유전자형 추론"""
        d = gene.dominant_allele
        r = gene.recessive_allele

        if gene.chromosome_type == ChromosomeType.X_LINKED:
            if parent_gender == Gender.MALE:
                # 아버지
                if child_gender == Gender.FEMALE:
                    # 딸에게 준 대립유전자
                    # child_genotype 중 하나가 아버지에게서 옴
                    possible = [child_genotype[0], child_genotype[1]]
                    allele = random.choice([a for a in possible if a])
                    return (allele, None)
                else:
                    # 아들에게는 Y만 줌 - 아버지 유전자형 무관
                    return random.choice([(d, None), (r, None)])
            else:
                # 어머니
                if child_gender == Gender.MALE:
                    # 아들이 받은 대립유전자
                    child_allele = child_genotype[0]
                    if child_allele.isupper():
                        return random.choice([(d, d), (d, r)])
                    else:
                        return random.choice([(d, r), (r, r)])
                else:
                    # 딸이 받은 것 중 하나
                    child_alleles = [child_genotype[0], child_genotype[1]]
                    # 어머니가 줄 수 있는 유전자형
                    has_d = any(a.isupper() for a in child_alleles if a)
                    has_r = any(a.islower() for a in child_alleles if a)
                    if has_d and has_r:
                        return (d, r)
                    elif has_d:
                        return random.choice([(d, d), (d, r)])
                    else:
                        return random.choice([(d, r), (r, r)])
        else:
            # 상염색체
            child_alleles = [child_genotype[0], child_genotype[1]]
            has_d = any(a.isupper() for a in child_alleles if a)
            has_r = any(a.islower() for a in child_alleles if a)

            if has_d and has_r:
                # 자녀가 이형접합 -> 부모가 둘 다 줄 수 있어야
                return (d, r)
            elif has_r:  # rr
                return random.choice([(d, r), (r, r)])
            else:  # DD 또는 Dd
                return random.choice([(d, d), (d, r)])

    def _assign_trait(
        self,
        person: Person,
        gene: Gene,
        genotype: Tuple[str, Optional[str]]
    ):
        """개인에게 유전 형질 할당"""
        trait = GeneticTrait(
            gene=gene,
            allele1=genotype[0],
            allele2=genotype[1]
        )
        person.add_trait(trait)

    def _validate_all_constraints(
        self,
        family: Family,
        constraints: List[PedigreeConstraint]
    ) -> bool:
        """모든 제약조건 만족 여부 확인"""
        for constraint in constraints:
            gene = constraint.gene

            if constraint.target_person_id:
                person = family.get_member(constraint.target_person_id)
                if person:
                    trait = person.get_trait(gene.symbol)
                    if trait:
                        if constraint.required_phenotype:
                            if trait.phenotype != constraint.required_phenotype:
                                return False
                        if constraint.required_carrier is not None:
                            if trait.is_carrier != constraint.required_carrier:
                                return False

            if constraint.min_affected_count > 0:
                gen = constraint.affected_in_generation or 2
                members = family.get_generation(gen)
                affected = sum(
                    1 for m in members
                    if m.get_trait(gene.symbol) and
                    m.get_trait(gene.symbol).phenotype == Phenotype.RECESSIVE
                )
                if affected < constraint.min_affected_count:
                    return False

        return True

    def generate_complete_problem(
        self,
        genes: Optional[List[Gene]] = None,
        mode: InheritanceMode = InheritanceMode.AUTOSOMAL_RECESSIVE,
        config: Optional[GenerationConfig] = None,
        min_affected: int = 1
    ) -> Tuple[bool, Family]:
        """
        완전한 문제 세트 생성

        Args:
            genes: 사용할 유전자 목록 (None이면 기본 생성)
            mode: 유전 방식
            config: 세대 설정
            min_affected: 최소 발병자 수

        Returns:
            (success, family)
        """
        if genes is None:
            # 기본 유전자 생성
            if mode in [InheritanceMode.X_LINKED_DOMINANT,
                       InheritanceMode.X_LINKED_RECESSIVE]:
                chromosome_type = ChromosomeType.X_LINKED
            else:
                chromosome_type = ChromosomeType.AUTOSOMAL

            genes = [
                Gene(
                    symbol='H',
                    dominant_allele='H',
                    recessive_allele='h',
                    chromosome_type=chromosome_type
                )
            ]

        # 가족 구조 생성
        family = self.generate_family_structure(config)

        # 제약조건 설정
        constraints = [
            PedigreeConstraint(
                gene=genes[0],
                min_affected_count=min_affected,
                affected_in_generation=2
            )
        ]

        # 역추적으로 유전자형 할당
        return self.assign_genotypes_with_backtracking(family, genes, constraints)

    def generate_multi_trait_problem(
        self,
        num_traits: int = 2,
        modes: Optional[List[InheritanceMode]] = None,
        config: Optional[GenerationConfig] = None
    ) -> Tuple[bool, Family]:
        """
        다인자 유전 문제 생성 (2개 이상의 형질)
        """
        if modes is None:
            modes = [
                InheritanceMode.AUTOSOMAL_RECESSIVE,
                InheritanceMode.X_LINKED_RECESSIVE
            ][:num_traits]

        genes = []
        gene_symbols = ['H', 'T', 'R', 'B']  # 형질별 기호

        for i, mode in enumerate(modes):
            if mode in [InheritanceMode.X_LINKED_DOMINANT,
                       InheritanceMode.X_LINKED_RECESSIVE]:
                chromosome_type = ChromosomeType.X_LINKED
            else:
                chromosome_type = ChromosomeType.AUTOSOMAL

            genes.append(Gene(
                symbol=gene_symbols[i],
                dominant_allele=gene_symbols[i].upper(),
                recessive_allele=gene_symbols[i].lower(),
                chromosome_type=chromosome_type
            ))

        # 가족 구조 생성
        family = self.generate_family_structure(config)

        # 각 형질별 제약조건
        constraints = []
        for gene in genes:
            constraints.append(PedigreeConstraint(
                gene=gene,
                min_affected_count=1,
                affected_in_generation=2
            ))

        return self.assign_genotypes_with_backtracking(family, genes, constraints)
