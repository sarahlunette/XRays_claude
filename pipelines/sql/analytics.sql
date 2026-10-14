-- Analytics views over the relational copy of the annotation store (revision plan 6.5).

-- 1. Per site, share of 2025 studies with at least one positive fracture label.
WITH study_pos AS (
  SELECT s.site_id, s.study_id,
         MAX(CASE WHEN l.label = 'fracture' AND l.value = 1 THEN 1 ELSE 0 END) AS has_fracture
  FROM studies s
  LEFT JOIN labels l ON l.study_id = s.study_id
  WHERE s.study_date >= '2025-01-01' AND s.study_date < '2026-01-01'
  GROUP BY s.site_id, s.study_id
)
SELECT site_id, COUNT(*) AS n_studies, AVG(has_fracture * 1.0) AS share_with_fracture
FROM study_pos
GROUP BY site_id
ORDER BY share_with_fracture DESC;

-- 2. Studies never labelled (anti-join): the annotation backlog.
SELECT s.study_id, s.site_id, s.study_date
FROM studies s
WHERE NOT EXISTS (SELECT 1 FROM labels l WHERE l.study_id = s.study_id);

-- 3. First study per patient (window function, rows are not collapsed like GROUP BY).
WITH ranked AS (
  SELECT s.*, ROW_NUMBER() OVER (PARTITION BY patient_id ORDER BY study_date) AS rn
  FROM studies s
)
SELECT * FROM ranked WHERE rn = 1;

-- 4. Reader disagreement: studies where two readers gave different fracture labels.
SELECT a.study_id, a.reader_id AS reader_a, b.reader_id AS reader_b, a.value AS value_a, b.value AS value_b
FROM labels a
JOIN labels b ON a.study_id = b.study_id AND a.label = b.label AND a.reader_id < b.reader_id
WHERE a.label = 'fracture' AND a.value <> b.value;

-- 5. Weekly positive rate per site with a 4-week moving average (output-drift dashboard).
SELECT site_id, week,
       positive_rate,
       AVG(positive_rate) OVER (PARTITION BY site_id ORDER BY week ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS ma4
FROM (
  SELECT site_id, DATE_TRUNC('week', ts) AS week, AVG(CASE WHEN positive THEN 1.0 ELSE 0.0 END) AS positive_rate
  FROM predictions
  GROUP BY site_id, DATE_TRUNC('week', ts)
) w;
