"""Versioned canonical briefs used by grounded authoring and generation."""

import hashlib
import json

from pydantic import BaseModel, Field


GROUNDING_BRIEF_VERSION = "pilot-grounding-v1"


class CanonicalGroundingBrief(BaseModel):
    skill_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    statements: list[str] = Field(min_length=1)
    intent_reference_ids: dict[str, list[str]] = Field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


PILOT_GROUNDING_BRIEFS = {
    brief.skill_id: brief
    for brief in (
        CanonicalGroundingBrief(
            skill_id="AI-FND-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "Intelligent-system capabilities include problem solving, reasoning, decision making, and learning.",
                "Concrete AI applications include face recognition, game playing, speech processing, and route finding.",
                "Automation alone does not establish that a system is intelligent.",
                "An intelligent agent receives percepts from its environment and selects actions based on them.",
                "Introductory questions should test recognition or simple interpretation rather than formal notation or full-list recall.",
            ],
            intent_reference_ids={
                "AI-FND-01-INT-01": [
                    "AI-FND-01-8bbbddaf2aa6",
                    "AI-FND-01-b50c85fa00a5",
                ],
                "AI-FND-01-INT-02": ["AI-FND-01-d03d77e0aca2"],
                "AI-FND-01-INT-03": ["AI-FND-01-f7c5eb1ccf76"],
            },
        ),
        CanonicalGroundingBrief(
            skill_id="AI-AGT-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "An agent perceives its environment and acts upon that environment.",
                "Sensors supply percepts or information from the environment; actuators carry out actions that affect it.",
                "PEAS means performance measure, environment, actuators, and sensors.",
                "A partially observable environment does not provide the agent with full state information.",
                "Questions must not treat the performance measure, environment, actuator, and sensor as interchangeable roles.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="AI-SRC-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "Initial state and start state are synonyms.",
                "An empty assignment has no assigned variables.",
                "Do not invent actor relationships or search states not stated in the sources.",
                "Action cost is the cost of an individual action; path cost is accumulated over a path, and the two must not be conflated.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="AI-SRC-02",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "A search tree is at least as large as its state-space graph.",
                "Cycles may make the search tree infinitely deep.",
                "Cycles do not imply that a goal is unreachable.",
                "Search-tree nodes distinguish paths, so a state may occur repeatedly.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="AI-SRC-03",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "The frontier contains generated search nodes that are waiting to be selected for expansion.",
                "Expansion removes a selected node, applies the available actions through the result model, and generates child nodes.",
                "In the project solvers, reached means discovered: BFS and DFS record a state when it is admitted to the frontier.",
                "Explored or expanded means a state has been removed from the frontier and expanded; these names must not be used as silent synonyms for reached.",
                "A repeated state is not added again when the existing record is already at least as good.",
                "For an unexpanded state in a cost-sensitive frontier, a newly discovered lower path cost replaces the stored frontier cost and parent record.",
                "The current solver does not reopen a state after it has entered the expanded set, so questions must not imply that it does.",
                "Questions for this skill explain frontier, reached-state, duplicate, and expansion mechanics without asking for a BFS, DFS, UCS, Greedy, or A-star expansion trace.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="AI-SRC-08",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "h(n) estimates the remaining forward cost.",
                "g(n) is the accumulated cost from the start.",
                "f(n) = g(n) + h(n).",
                "Greedy Best-First Search prioritizes h(n).",
                "UCS and Dijkstra's algorithm prioritize g(n).",
                "Manhattan distance must only be used for an appropriate grid setting.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="DSA-CPX-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "Asymptotic (Big-O) notation describes how an algorithm's running time grows as input size increases, not its exact running time.",
                "Changing a constant factor in a running-time expression shifts where two growth curves cross, but does not change which one eventually grows faster.",
                "A faster computer or a constant-factor code optimization does not change an algorithm's asymptotic growth rate.",
                "Linear search is O(n); binary search on sorted data is O(log n) and is more efficient for large inputs.",
                "Binary search requires the data to already be sorted; it does not work correctly on unsorted data.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="DSA-LST-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "An array stores its elements in contiguous memory, which allows any element to be read directly by index in constant time.",
                "Inserting into or deleting from the middle of an array-based list requires shifting the surrounding elements.",
                "A linked-list node stores an element together with a pointer (the next field) to the following node, not a copy of the following element's value.",
                "A singly linked list must be traversed node by node from the head to reach a given position; it does not support constant-time indexed access like an array.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="DSA-STK-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "A stack is Last-In, First-Out (LIFO): elements are pushed onto and popped from the same end, called the top.",
                "A queue is First-In, First-Out (FIFO): elements are enqueued at the back and dequeued from the front.",
                "A stack pop always returns the most recently pushed remaining element, never the earliest one.",
                "A queue dequeue always returns the earliest remaining enqueued element, never the most recent one.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="DSA-SRC-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "Linear search examines elements one at a time, in order, until it finds the target or exhausts the list; its running time is O(n).",
                "Binary search only works on sorted data: it compares the target to the middle element and continues searching only the half of the list that could still contain the target.",
                "Binary search's running time is O(log n), which is more efficient than linear search's O(n) for large sorted inputs.",
                "Binary search must discard, not continue searching, the half of the list that cannot contain the target.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="DSA-SRT-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "Bubble sort repeatedly compares adjacent elements and swaps them if out of order; one pass does not fully sort the list.",
                "Selection sort repeatedly finds the smallest element in the unsorted portion and moves it to the front of that portion.",
                "Insertion sort inserts each next element into its correct position among the already-sorted elements that precede it.",
                "Merge sort is divide-and-conquer: it splits the list in half, recursively sorts each half, then merges the two sorted halves.",
                "Quicksort is divide-and-conquer: it selects a pivot, partitions the remaining elements into those less than and greater than the pivot, and recursively sorts each partition -- it does not split the list at a fixed midpoint the way merge sort does.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="DSA-HSH-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "A hash function maps a search key to a position (index) in the hash table; it does not sort the table.",
                "A collision occurs when two different keys hash to the same table position; a collision is a normal, expected event, not a hash-function failure.",
                "Under chaining, a colliding key is appended to the linked list already stored at that index rather than overwriting the existing entry.",
                "Under open addressing, when a key's home position is already occupied, the collision resolution policy probes a sequence of other slots until it finds a free one, rather than failing the insertion.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="DSA-TGR-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "In a binary search tree, every node in the left subtree of a node with key K has a key less than or equal to K, and every node in the right subtree has a key greater than K.",
                "An inorder traversal of a binary search tree visits its nodes in ascending sorted order.",
                "In a max-heap, every node's value is greater than or equal to the values of its children; a heap is not ordered the way a binary search tree is.",
                "A graph traversal (such as breadth-first or depth-first search) visits every vertex exactly once; a visiting order that skips or repeats a vertex is not a valid traversal.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="LA-SLE-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "A matrix is in reduced row-echelon form (RREF) when the leading entry of each nonzero row is a 1, that leading 1 is the only nonzero entry in its column, and each leading 1 sits to the right of every leading 1 in the rows above it.",
                "Course convention: when a question asks to explicitly solve for the value of a variable, use RREF and read the solution directly off the leading-1 columns -- do not reason through back-substitution.",
                "Course convention: a conclusion drawn only from row-echelon form (REF), such as whether the system is consistent from a zero row, is a structural conclusion and must be distinguished from finding the actual RREF solution.",
                "Solving a system of linear equations means finding all values of the variables that make every equation in the system simultaneously true, not just one equation.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="LA-VSP-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "The span of a set of vectors is the set of all possible linear combinations of those vectors.",
                "A set of vectors is linearly independent exactly when the only linear combination of them equal to the zero vector uses all-zero coefficients (the trivial relation); any nontrivial combination equal to zero makes the set linearly dependent.",
                "A basis of a vector space is a set that both spans the space and is linearly independent; a spanning-only or independent-only set is not automatically a basis.",
                "A subset of a vector space is a subspace when it is nonempty and closed under vector addition and scalar multiplication; containing the zero vector alone is not sufficient.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="LA-LTR-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "A linear transformation T must satisfy both T(u1 + u2) = T(u1) + T(u2) for all inputs, and T(alpha*u) = alpha*T(u) for every scalar alpha; a function is not linear if it satisfies only one property.",
                "A function that adds a nonzero constant, or that includes a nonlinear term, is not a linear transformation even if it otherwise resembles a matrix-vector product.",
                "The standard matrix A of a linear transformation T on column vectors is built so that T(x) = A*x for every input x; its columns are the images of the standard basis vectors, not its rows.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="LA-DET-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "A square matrix is nonsingular (invertible) exactly when its determinant is nonzero; a zero determinant means the matrix has no inverse.",
                "A matrix with nonzero determinant has only the trivial (zero) solution to its associated homogeneous system.",
                "Cramer's Rule solves for one variable at a time by replacing that variable's column in the coefficient matrix with the constant column, then dividing the resulting determinant by the coefficient matrix's own determinant.",
                "Cramer's Rule only gives a unique solution when the coefficient matrix's determinant is nonzero; when it is zero, Cramer's Rule cannot be used.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="LA-EIG-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "A nonzero vector x is an eigenvector of a square matrix A with eigenvalue lambda exactly when A*x = lambda*x; the zero vector is never counted as an eigenvector.",
                "The eigenvalues of a matrix are exactly the roots of its characteristic polynomial.",
                "For a given eigenvalue, the set of its eigenvectors together with the zero vector forms a subspace called the eigenspace for that eigenvalue.",
                "Two square matrices A and B are similar when A = S*B*S^-1 for some nonsingular matrix S; similar matrices always share the same eigenvalues, even though they are not equal as matrices.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="LA-MKV-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "A square matrix is stochastic when all of its entries are nonnegative and the entries of each column sum to 1.",
                "A Markov chain is a difference equation v_(t+1) = A*v_t driven by a stochastic matrix A, where v_(t+1) is the state one time step after v_t, not the same state.",
                "Google's PageRank Importance Rule: if a page links to n other pages, each of those pages inherits 1/n of the linking page's importance, not all of it.",
                "The PageRank vector is the steady-state vector of the importance matrix's Markov chain -- an eigenvector of the (stochastic) importance matrix with eigenvalue 1 -- not simply the matrix's row or column of largest entries.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="DB-ERM-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "In Chen notation, entity types are drawn as rectangles, relationship types as diamonds, and attributes as ovals connected by a line to the entity or relationship they describe.",
                "Cardinality symbols (such as 1 and n) label the lines connecting a relationship diamond to its related entities, not the entities or attributes themselves.",
                "An entity-relationship diagram gives an overview of a database's design -- its entities, attributes, and relationships -- built through an iterative modelling process, not SQL code or a physical storage layout.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="DB-REL-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "In the relational model, a row of a relation (table) is called a tuple (or record), and a column is called an attribute (or field); a domain is the set of atomic values an attribute may take.",
                "Entity integrity requires that every table have a primary key whose value is unique across rows and never null.",
                "Referential integrity requires that a foreign-key value either matches an existing primary-key value in the referenced table, or is null; a foreign key matching no primary key violates referential integrity.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="DB-ALG-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "The relational-algebra selection operator (sigma) filters the rows of a relation by a condition, keeping the same schema (same columns); it is equivalent to SQL's WHERE clause, not its SELECT clause.",
                "The relational-algebra projection operator (pi) changes the schema by keeping only the specified columns; it does not filter rows by a condition the way selection does.",
                "A single SQL query can correspond to more than one equivalent relational-algebra expression -- for example, applying selection before or after projection can give the same result -- as long as projection does not remove a column that selection still needs.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="DB-SQL-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "SQL's core DML statements are SELECT (query data), INSERT (add rows), UPDATE (modify existing rows), and DELETE (remove rows); each performs a distinct task.",
                "An inner join returns only the rows that have matching values in both joined tables; it does not include unmatched rows from either table.",
                "A SELECT statement's WHERE clause filters which rows are returned; it does not determine which columns appear in the output.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="DB-NRM-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "A functional dependency X -> Y means that for every valid instance of X, that value of X uniquely determines the value of Y; X is the determinant and Y is the dependent, and the relationship is not symmetric.",
                "A functional dependency X -> Y is violated by any two rows that share the same X value but have different Y values.",
                "Normalization is the process of determining and reducing redundancy in a relational schema, not a process for encrypting or securing data.",
                "A relation is in third normal form (3NF) only if it is already in second normal form and contains no transitive dependency -- no non-key attribute depending on another non-key attribute rather than directly on the primary key.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="DB-TXN-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "A transaction is a set of database operations treated as one undividable unit of work: it either succeeds and commits in its entirety, or fails and none of its operations take effect.",
                "ACID stands for atomicity (all-or-nothing execution), consistency (only valid data is written), isolation (concurrent transactions do not see each other's incomplete effects), and durability (committed changes persist even after a failure).",
                "Concurrency control coordinates transactions executing simultaneously on the same data so that mutual interference does not cause inconsistencies; it is not primarily about making transactions run faster.",
                "When a transaction fails before committing, the transaction manager rolls back its partial changes using a log file, so none of those changes remain in the database.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="DB-IDX-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "A B-tree generalizes a binary search tree: instead of one value, each node holds a list of multiple values, which lets an index skip most of the table instead of scanning every row.",
                "In a B+ tree, record data is stored only in the leaf nodes; internal nodes hold only keys and pointers used to guide the search, not the record data itself.",
                "A B+ tree's wider fan-out per node keeps the tree shorter than an equivalent binary search tree over the same keys, so fewer levels must be traversed to reach a given key.",
            ],
        ),
    )
}


def grounding_brief(skill_id: str) -> CanonicalGroundingBrief:
    try:
        return PILOT_GROUNDING_BRIEFS[skill_id]
    except KeyError as error:
        raise ValueError(f"no canonical grounding brief for {skill_id}") from error
