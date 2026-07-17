# Metadata graph-readiness audit

**Files audited:** 36  |  **ERROR:** 1  |  **WARN:** 0  |  **INFO:** 0


## T01-complete.json — PASS  (trial `T01`, not_eligible)

- segments: 11, boxes: 13, AOIs: 10, star segments: 0, doc image: T01.png 1920x1080
- segment types: {'sentence': 9, 'mark_scheme_point': 2}
- mark_point_id coverage (mark-scheme bullets labelled): 2/2 (metadata completeness only)

No issues found.


## T02-complete.json — PASS  (trial `T02`, not_eligible)

- segments: 4, boxes: 5, AOIs: 10, star segments: 0, doc image: T02.png 1920x1080
- segment types: {'sentence': 3, 'mark_scheme_point': 1}
- mark_point_id coverage (mark-scheme bullets labelled): 1/1 (metadata completeness only)

No issues found.


## T03-complete.json — PASS  (trial `T03`, not_eligible)

- segments: 27, boxes: 41, AOIs: 11, star segments: 0, doc image: T03.png 1920x1080
- segment types: {'sentence': 7, 'bullet_point': 5, 'clause': 1, 'mark_scheme_point': 8, 'commentary_guidance': 6}
- mark_point_id coverage (mark-scheme bullets labelled): 8/8 (metadata completeness only)

No issues found.


## T04-complete.json — PASS  (trial `T04`, not_eligible)

- segments: 14, boxes: 24, AOIs: 11, star segments: 0, doc image: T04.png 1920x1080
- segment types: {'sentence': 6, 'mark_scheme_point': 3, 'commentary_guidance': 5}
- mark_point_id coverage (mark-scheme bullets labelled): 3/3 (metadata completeness only)

No issues found.


## T05-complete.json — PASS  (trial `T05`, not_eligible)

- segments: 15, boxes: 21, AOIs: 11, star segments: 0, doc image: T05.png 1920x1080
- segment types: {'sentence': 5, 'bullet_point': 2, 'clause': 1, 'mark_scheme_point': 3, 'commentary_guidance': 4}
- mark_point_id coverage (mark-scheme bullets labelled): 3/3 (metadata completeness only)

No issues found.


## T06-complete.json — PASS  (trial `T06`, not_eligible)

- segments: 8, boxes: 13, AOIs: 11, star segments: 0, doc image: T06.png 1920x1080
- segment types: {'sentence': 5, 'mark_scheme_point': 1, 'commentary_guidance': 2}
- mark_point_id coverage (mark-scheme bullets labelled): 1/1 (metadata completeness only)

No issues found.


## T07-complete.json — PASS  (trial `T07`, not_eligible)

- segments: 16, boxes: 32, AOIs: 11, star segments: 0, doc image: T07.png 1920x1080
- segment types: {'sentence': 7, 'mark_scheme_point': 2, 'commentary_guidance': 7}
- mark_point_id coverage (mark-scheme bullets labelled): 2/2 (metadata completeness only)

No issues found.


## T08-complete.json — PASS  (trial `T08`, not_eligible)

- segments: 23, boxes: 49, AOIs: 11, star segments: 0, doc image: T08.png 1920x1080
- segment types: {'sentence': 8, 'mark_scheme_point': 4, 'commentary_guidance': 11}
- mark_point_id coverage (mark-scheme bullets labelled): 4/4 (metadata completeness only)

No issues found.


## T09-complete.json — PASS  (trial `T09`, not_eligible)

- segments: 33, boxes: 68, AOIs: 11, star segments: 0, doc image: T09.png 1920x1080
- segment types: {'sentence': 11, 'clause': 2, 'mark_scheme_point': 11, 'commentary_guidance': 9}
- mark_point_id coverage (mark-scheme bullets labelled): 11/11 (metadata completeness only)

No issues found.


## T10-completee.json — PASS  (trial `T10`, not_eligible)

- segments: 20, boxes: 38, AOIs: 11, star segments: 0, doc image: T10.png 1920x1080
- segment types: {'sentence': 6, 'clause': 3, 'mark_scheme_point': 4, 'commentary_guidance': 7}
- mark_point_id coverage (mark-scheme bullets labelled): 4/4 (metadata completeness only)

No issues found.


## T11NS-complete.json — FAIL  (trial `T11`, star_off)

- segments: 56, boxes: 80, AOIs: 12, star segments: 0, doc image: T11NS.png 1920x1080
- segment types: {'sentence': 4, 'clause': 18, 'bullet_point': 2, 'level_descriptor': 4, 'mark_scheme_point': 12, 'commentary_guidance': 16}
- mark_point_id coverage (mark-scheme bullets labelled): 12/12 (metadata completeness only)

### ERROR (1)

| category | entity | id | message | suggested fix |
|---|---|---|---|---|
| no_text | segment | `ann_segment_060` | Both corrected_text and ocr_text empty; node has no text embedding source. | Add the segment text. |


## T11S-complete.json — PASS  (trial `T11`, star_on)

- segments: 67, boxes: 92, AOIs: 13, star segments: 8, doc image: T11S.png 1920x1301
- segment types: {'sentence': 5, 'clause': 18, 'bullet_point': 2, 'level_descriptor': 4, 'mark_scheme_point': 12, 'commentary_guidance': 18, 'star_concept': 8}
- mark_point_id coverage (mark-scheme bullets labelled): 12/12 (metadata completeness only)

No issues found.


## T12NS-complete.json — PASS  (trial `T12`, star_off)

- segments: 53, boxes: 86, AOIs: 12, star segments: 0, doc image: T12NS.png 1920x1080
- segment types: {'sentence': 4, 'clause': 4, 'bullet_point': 8, 'level_descriptor': 4, 'mark_scheme_point': 10, 'commentary_guidance': 23}
- mark_point_id coverage (mark-scheme bullets labelled): 10/10 (metadata completeness only)

No issues found.


## T12S-complete.json — PASS  (trial `T12`, star_on)

- segments: 69, boxes: 107, AOIs: 13, star segments: 12, doc image: T12S.png 1920x1544
- segment types: {'sentence': 4, 'clause': 4, 'bullet_point': 8, 'level_descriptor': 4, 'mark_scheme_point': 10, 'commentary_guidance': 27, 'star_concept': 12}
- mark_point_id coverage (mark-scheme bullets labelled): 10/10 (metadata completeness only)

No issues found.


## T13NS-complete.json — PASS  (trial `T13`, star_off)

- segments: 36, boxes: 55, AOIs: 12, star segments: 0, doc image: T13NS.png 1920x1080
- segment types: {'sentence': 5, 'bullet_point': 9, 'clause': 2, 'level_descriptor': 3, 'mark_scheme_point': 7, 'commentary_guidance': 10}
- mark_point_id coverage (mark-scheme bullets labelled): 7/7 (metadata completeness only)

No issues found.


## T13S-complete.json — PASS  (trial `T13`, star_on)

- segments: 56, boxes: 76, AOIs: 13, star segments: 11, doc image: T13S.png 1920x1173
- segment types: {'sentence': 12, 'bullet_point': 9, 'clause': 2, 'level_descriptor': 3, 'mark_scheme_point': 7, 'commentary_guidance': 12, 'star_concept': 11}
- mark_point_id coverage (mark-scheme bullets labelled): 7/7 (metadata completeness only)

No issues found.


## T14-complete.json — PASS  (trial `T14`, not_eligible)

- segments: 39, boxes: 79, AOIs: 12, star segments: 0, doc image: T14.png 1920x1080
- segment types: {'sentence': 3, 'clause': 2, 'bullet_point': 6, 'level_descriptor': 4, 'mark_scheme_point': 11, 'commentary_guidance': 13}
- mark_point_id coverage (mark-scheme bullets labelled): 11/11 (metadata completeness only)

No issues found.


## T15-complete.json — PASS  (trial `T15`, not_eligible)

- segments: 31, boxes: 59, AOIs: 11, star segments: 0, doc image: T15.png 1920x1080
- segment types: {'sentence': 5, 'clause': 5, 'level_descriptor': 2, 'mark_scheme_point': 7, 'commentary_guidance': 12}
- mark_point_id coverage (mark-scheme bullets labelled): 7/7 (metadata completeness only)

No issues found.


## T16-complete.json — PASS  (trial `T16`, not_eligible)

- segments: 41, boxes: 87, AOIs: 12, star segments: 0, doc image: T16.png 1920x1080
- segment types: {'sentence': 13, 'clause': 1, 'bullet_point': 1, 'level_descriptor': 4, 'mark_scheme_point': 7, 'commentary_guidance': 15}
- mark_point_id coverage (mark-scheme bullets labelled): 7/7 (metadata completeness only)

No issues found.


## T17-complete.json — PASS  (trial `T17`, not_eligible)

- segments: 25, boxes: 48, AOIs: 11, star segments: 0, doc image: T17.png 1920x1080
- segment types: {'sentence': 11, 'clause': 1, 'mark_scheme_point': 3, 'commentary_guidance': 10}
- mark_point_id coverage (mark-scheme bullets labelled): 3/3 (metadata completeness only)

No issues found.


## T18-complete.json — PASS  (trial `T18`, not_eligible)

- segments: 15, boxes: 29, AOIs: 11, star segments: 0, doc image: T18.png 1920x1080
- segment types: {'sentence': 7, 'mark_scheme_point': 4, 'commentary_guidance': 4}
- mark_point_id coverage (mark-scheme bullets labelled): 4/4 (metadata completeness only)

No issues found.


## T19-complete.json — PASS  (trial `T19`, not_eligible)

- segments: 12, boxes: 15, AOIs: 9, star segments: 0, doc image: T19.png 1920x1080
- segment types: {'sentence': 10, 'mark_scheme_point': 2}
- mark_point_id coverage (mark-scheme bullets labelled): 2/2 (metadata completeness only)

No issues found.


## T20-complete.json — PASS  (trial `T20`, not_eligible)

- segments: 24, boxes: 38, AOIs: 11, star segments: 0, doc image: T20.png 1920x1080
- segment types: {'sentence': 10, 'bullet_point': 2, 'mark_scheme_point': 3, 'commentary_guidance': 9}
- mark_point_id coverage (mark-scheme bullets labelled): 3/3 (metadata completeness only)

No issues found.


## T21NS-complete.json — PASS  (trial `T21`, star_off)

- segments: 48, boxes: 81, AOIs: 12, star segments: 0, doc image: T21NS.png 1920x1080
- segment types: {'clause': 15, 'bullet_point': 2, 'sentence': 3, 'level_descriptor': 3, 'mark_scheme_point': 8, 'commentary_guidance': 17}
- mark_point_id coverage (mark-scheme bullets labelled): 8/8 (metadata completeness only)

No issues found.


## T21S-complete.json — PASS  (trial `T21`, star_on)

- segments: 57, boxes: 92, AOIs: 13, star segments: 6, doc image: T21S.png 1920x1193
- segment types: {'clause': 15, 'bullet_point': 2, 'sentence': 4, 'level_descriptor': 3, 'mark_scheme_point': 8, 'commentary_guidance': 19, 'star_concept': 6}
- mark_point_id coverage (mark-scheme bullets labelled): 8/8 (metadata completeness only)

No issues found.


## T22-complete.json — PASS  (trial `T22`, not_eligible)

- segments: 19, boxes: 34, AOIs: 11, star segments: 0, doc image: T22.png 1920x1080
- segment types: {'sentence': 9, 'clause': 2, 'mark_scheme_point': 3, 'commentary_guidance': 5}
- mark_point_id coverage (mark-scheme bullets labelled): 3/3 (metadata completeness only)

No issues found.


## T23-complete.json — PASS  (trial `T23`, not_eligible)

- segments: 19, boxes: 36, AOIs: 10, star segments: 0, doc image: T23.png 1920x1080
- segment types: {'sentence': 8, 'clause': 3, 'mark_scheme_point': 3, 'commentary_guidance': 5}
- mark_point_id coverage (mark-scheme bullets labelled): 3/3 (metadata completeness only)

No issues found.


## T24-complete.json — PASS  (trial `T24`, not_eligible)

- segments: 35, boxes: 72, AOIs: 11, star segments: 0, doc image: T24.png 1920x1080
- segment types: {'sentence': 9, 'bullet_point': 5, 'mark_scheme_point': 8, 'commentary_guidance': 13}
- mark_point_id coverage (mark-scheme bullets labelled): 8/8 (metadata completeness only)

No issues found.


## T25-complete.json — PASS  (trial `T25`, not_eligible)

- segments: 12, boxes: 13, AOIs: 10, star segments: 0, doc image: T25.png 1920x1080
- segment types: {'sentence': 10, 'mark_scheme_point': 2}
- mark_point_id coverage (mark-scheme bullets labelled): 2/2 (metadata completeness only)

No issues found.


## T26-complete.json — PASS  (trial `T26`, not_eligible)

- segments: 9, boxes: 10, AOIs: 11, star segments: 0, doc image: T26.png 1920x1080
- segment types: {'sentence': 5, 'mark_scheme_point': 2, 'commentary_guidance': 2}
- mark_point_id coverage (mark-scheme bullets labelled): 2/2 (metadata completeness only)

No issues found.


## T27NS-complete.json — PASS  (trial `T27`, star_off)

- segments: 47, boxes: 71, AOIs: 12, star segments: 0, doc image: T27NS.png 1920x1080
- segment types: {'sentence': 10, 'clause': 5, 'bullet_point': 2, 'level_descriptor': 3, 'mark_scheme_point': 13, 'commentary_guidance': 14}
- mark_point_id coverage (mark-scheme bullets labelled): 13/13 (metadata completeness only)

No issues found.


## T27S-complete.json — PASS  (trial `T27`, star_on)

- segments: 61, boxes: 93, AOIs: 13, star segments: 7, doc image: T27S.png 1920x1280
- segment types: {'sentence': 11, 'clause': 5, 'bullet_point': 2, 'level_descriptor': 3, 'mark_scheme_point': 13, 'commentary_guidance': 20, 'star_concept': 7}
- mark_point_id coverage (mark-scheme bullets labelled): 13/13 (metadata completeness only)

No issues found.


## T28-complete.json — PASS  (trial `T28`, not_eligible)

- segments: 25, boxes: 31, AOIs: 11, star segments: 0, doc image: T28.png 1920x1080
- segment types: {'sentence': 13, 'mark_scheme_point': 7, 'commentary_guidance': 5}
- mark_point_id coverage (mark-scheme bullets labelled): 7/7 (metadata completeness only)

No issues found.


## T29-complete.json — PASS  (trial `T29`, not_eligible)

- segments: 22, boxes: 47, AOIs: 11, star segments: 0, doc image: T29.png 1920x1080
- segment types: {'sentence': 7, 'clause': 1, 'mark_scheme_point': 4, 'commentary_guidance': 10}
- mark_point_id coverage (mark-scheme bullets labelled): 4/4 (metadata completeness only)

No issues found.


## T30NS-complete.json — PASS  (trial `T30`, star_off)

- segments: 61, boxes: 105, AOIs: 12, star segments: 0, doc image: T30NS.png 1920x1080
- segment types: {'sentence': 5, 'bullet_point': 6, 'level_descriptor': 5, 'mark_scheme_point': 23, 'commentary_guidance': 22}
- mark_point_id coverage (mark-scheme bullets labelled): 23/23 (metadata completeness only)

No issues found.


## T30S-complete.json — PASS  (trial `T30`, star_on)

- segments: 96, boxes: 154, AOIs: 13, star segments: 16, doc image: T30S.png 1920x2344
- segment types: {'sentence': 5, 'bullet_point': 6, 'level_descriptor': 5, 'mark_scheme_point': 23, 'commentary_guidance': 41, 'star_concept': 16}
- mark_point_id coverage (mark-scheme bullets labelled): 23/23 (metadata completeness only)

No issues found.
