// Annotation store (revision plan 6.4). Run with: mongosh azml pipelines/mongo/annotations.js

// Document: annotations embedded (read together, bounded size); dataset versions as an array.
db.images.insertOne({
  _id: "img_8f3a",
  study_uid: "1.2.840.0000", patient_hash: "p_19c2", site_id: "site_a",
  body_part: "WRIST", view: "PA", manufacturer: "SynthCo",
  storage: "s3://xrayml-lake/png16/img_8f3a.png",
  annotations: [{ reader: "rad_07", label: "fracture", box: [412, 880, 470, 945], date: "2026-03-02" }],
  dataset_versions: ["v12", "v13"],
});

// Compound index in ESR order: Equality (body_part, site_id), Sort (study_date), Range.
db.images.createIndex({ body_part: 1, site_id: 1, study_date: 1 });
db.images.createIndex({ dataset_versions: 1 });  // multikey index on the array

// Positive images per site and body part in dataset v13.
db.images.aggregate([
  { $match: { dataset_versions: "v13" } },
  { $unwind: "$annotations" },
  { $match: { "annotations.label": "fracture" } },
  { $group: { _id: { site: "$site_id", part: "$body_part" }, imgs: { $addToSet: "$_id" } } },
  { $project: { n_images: { $size: "$imgs" } } },
  { $sort: { n_images: -1 } },
]);

// Check the plan uses the index (IXSCAN, not COLLSCAN).
db.images.find({ body_part: "WRIST", site_id: "site_a" }).sort({ study_date: 1 }).explain("executionStats");
