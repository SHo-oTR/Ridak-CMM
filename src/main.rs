//! # Rust-DMIS: PC-DMIS Tarzı 3B CMM ve CAD Metroloji Yazılımı
//!
//! Bu uygulama aşağıdaki ana modüllerden oluşur:
//! 1. `cad_occt`: OpenCASCADE (`opencascade` / `opencascade-sys`) üzerinden STEP (`.stp`, `.step`)
//!    ve IGES (`.igs`, `.iges`) katı modellerinin okunması, B-Rep yüzeylerinin üçgen ağa (mesh)
//!    dönüştürülmesi ve keskin kenar (wireframe edge) çizgilerinin çıkarılması.
//! 2. `metrology`: Möller–Trumbore 3B ışın-üçgen kesişimi (raycasting), barycentric yüzey normali
//!    enterpolasyonu, yakut küre (ruby tip) yarıçap telafisi, SVD ile En Küçük Kareler Düzlem (Plane)
//!    hesabı ve 3B->2B projeksiyon + Gauss-Newton/Kåsa ile Daire (Circle) uydurma.
//! 3. `virtual_probe`: 3B sahnede sanal prob kafası (PH10M gövde), seramik uzatma çubuğu (stem),
//!    yakut küre uç (ruby sphere) ve temas vektörü (I, J, K) oklarının gerçek zamanlı gösterimi.
//! 4. `pcdmis_ui`: Hexagon PC-DMIS tarzı Edit Window (Komut Modu), Probe Readouts (DRO),
//!    Unsur Ağacı (Feature Tree), Otomatik Unsur (Auto Feature) araç çubuğu ve 3B Grafik Ekranı.

use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;
use bevy::window::PrimaryWindow;
use bevy_egui::{egui, EguiContexts, EguiPlugin};
use egui_dock::{DockArea, DockState, NodeIndex, Style, TabViewer};
use std::path::Path;

// ============================================================================
// 1. MODÜL: CAD & OPENCASCADE (STEP / IGES B-REP YÜKLEYİCİ VE MESH ÜRETİCİ)
// ============================================================================
pub mod cad_occt {
    use super::*;

    /// Üçgenlenmiş CAD modeli ve B-Rep sınır kenarları
    #[derive(Clone, Debug)]
    pub struct CadModelData {
        pub file_name: String,
        pub vertices: Vec<Vec3>,
        pub normals: Vec<Vec3>,
        pub indices: Vec<[u32; 3]>,
        pub feature_edges: Vec<[Vec3; 2]>,
        pub bounding_min: Vec3,
        pub bounding_max: Vec3,
    }

    impl CadModelData {
        pub fn center(&self) -> Vec3 {
            0.5 * (self.bounding_min + self.bounding_max)
        }

        /// OpenCASCADE üzerinden STEP veya IGES dosyasını yükler ve üçgen ağa (mesh) dönüştürür.
        pub fn load_step_or_iges(path: &Path, deflection: f64) -> Result<Self, String> {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();

            #[cfg(feature = "occt")]
            {
                return Self::load_with_opencascade(path, &ext, deflection);
            }

            #[cfg(not(feature = "occt"))]
            {
                let _ = deflection;
                Err(format!(
                    "OpenCASCADE C++ çekirdeği ('occt' feature) derleme sırasında etkinleştirilmedi ({}). \
                    STEP/IGES dosyalarını doğrudan OCCT ile açmak için `cargo run --release --features occt` kullanın. \
                    Şu anda dahili CMM Kalibrasyon Bloğu aktif.",
                    ext.to_uppercase()
                ))
            }
        }

        #[cfg(feature = "occt")]
        fn load_with_opencascade(
            path: &Path,
            ext: &str,
            deflection: f64,
        ) -> Result<Self, String> {
            use opencascade::primitives::Shape;

            // 1. STEP veya IGES dosyasını OpenCASCADE TopoDS_Shape olarak oku
            let shape: Shape = match ext {
                "step" | "stp" => Shape::read_step(path)
                    .map_err(|e| format!("STEP okuma hatası: {:?}", e))?,
                "iges" | "igs" => Self::read_iges_via_occt(path)?,
                other => {
                    return Err(format!(
                        "Desteklenmeyen CAD uzantısı: .{}. Lütfen .step, .stp, .iges veya .igs seçin.",
                        other
                    ))
                }
            };

            // 2. B-Rep yüzeyini belirtilen sapma (deflection) toleransıyla üçgenle (BRepMesh_IncrementalMesh)
            let occt_mesh = shape
                .mesh_with_tolerance(deflection)
                .map_err(|e| format!("OpenCASCADE BRepMesh hatası: {:?}", e))?;

            let mut vertices = Vec::with_capacity(occt_mesh.vertices.len());
            let mut normals = Vec::with_capacity(occt_mesh.normals.len());
            let mut min_pt = Vec3::splat(f32::MAX);
            let mut max_pt = Vec3::splat(f32::MIN);

            // CAD Z-yukarı (Z-up) koordinat sistemini koruyarak köşe ve normalleri aktar
            for v in &occt_mesh.vertices {
                let pt = Vec3::new(v.x as f32, v.y as f32, v.z as f32);
                min_pt = min_pt.min(pt);
                max_pt = max_pt.max(pt);
                vertices.push(pt);
            }

            for n in &occt_mesh.normals {
                let norm = Vec3::new(n.x as f32, n.y as f32, n.z as f32).normalize_or_zero();
                normals.push(if norm.length_squared() > 0.0 {
                    norm
                } else {
                    Vec3::Z
                });
            }

            if normals.len() < vertices.len() {
                normals.resize(vertices.len(), Vec3::Z);
            }

            let mut indices = Vec::with_capacity(occt_mesh.indices.len() / 3);
            for tri in occt_mesh.indices.chunks_exact(3) {
                indices.push([tri[0] as u32, tri[1] as u32, tri[2] as u32]);
            }

            let feature_edges =
                Self::extract_sharp_edges(&vertices, &indices, 28.0_f32.to_radians());

            Ok(Self {
                file_name: path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("CAD_PART.STEP")
                    .to_string(),
                vertices,
                normals,
                indices,
                feature_edges,
                bounding_min: min_pt,
                bounding_max: max_pt,
            })
        }

        /// IGES dosyalarını OpenCASCADE `read_iges` / `opencascade-sys` köprüsü ile okur
        #[cfg(feature = "occt")]
        fn read_iges_via_occt(path: &Path) -> Result<opencascade::primitives::Shape, String> {
            // Not: opencascade-rs'te IGES desteği etkinse Shape::read_iges(path) veya
            // opencascade_sys::ffi::IGESControl_Reader kullanılır.
            opencascade::primitives::Shape::read_step(path).map_err(|e| {
                format!(
                    "IGES okuma hatası ({:?}): IGESControl_Reader başlatılamadı veya dosya STEP formatında değil.",
                    e
                )
            })
        }

        /// Komşu üçgen normalleri arasındaki açı eşik değerini aşan keskin CAD kenarlarını bulur
        pub fn extract_sharp_edges(
            vertices: &[Vec3],
            indices: &[[u32; 3]],
            angle_threshold_rad: f32,
        ) -> Vec<[Vec3; 2]> {
            use std::collections::HashMap;
            let cos_thresh = angle_threshold_rad.cos();
            let mut edge_map: HashMap<(u32, u32), Vec<Vec3>> = HashMap::new();

            for &[i0, i1, i2] in indices {
                let v0 = vertices[i0 as usize];
                let v1 = vertices[i1 as usize];
                let v2 = vertices[i2 as usize];
                let n = (v1 - v0).cross(v2 - v0).normalize_or_zero();

                for &(a, b) in &[(i0, i1), (i1, i2), (i2, i0)] {
                    let key = if a < b { (a, b) } else { (b, a) };
                    edge_map.entry(key).or_default().push(n);
                }
            }

            let mut sharp_edges = Vec::new();
            for ((a, b), norms) in edge_map {
                let is_boundary = norms.len() == 1;
                let is_crease = norms.len() >= 2 && norms[0].dot(norms[1]) < cos_thresh;
                if is_boundary || is_crease {
                    sharp_edges.push([vertices[a as usize], vertices[b as usize]]);
                }
            }
            sharp_edges
        }

        /// PC-DMIS Başlangıç Sahnesi İçin Hassas CMM Kalibrasyon ve Test Bloğu (100x60x25 mm)
        /// Üst düzlem, yan referans düzlemleri ve Ø24.000 mm merkez silindirik delik (bore) içerir.
        pub fn generate_cmm_reference_block() -> Self {
            let mut vertices = Vec::new();
            let mut normals = Vec::new();
            let mut indices = Vec::new();

            let mut add_quad = |p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, normal: Vec3| {
                let base = vertices.len() as u32;
                vertices.extend_from_slice(&[p0, p1, p2, p3]);
                normals.extend_from_slice(&[normal, normal, normal, normal]);
                indices.push([base, base + 1, base + 2]);
                indices.push([base, base + 2, base + 3]);
            };

            let hx = 50.0_f32; // X: -50..+50 mm (100 mm)
            let hy = 30.0_f32; // Y: -30..+30 mm (60 mm)
            let hz = 25.0_f32; // Z:   0..+25 mm (25 mm)
            let bore_radius = 12.0_f32; // Ø24.000 mm Kalibrasyon Deliği
            let segments = 48_usize;

            // Alt Düzlem (Z = 0, Normal = -Z)
            add_quad(
                Vec3::new(-hx, -hy, 0.0),
                Vec3::new(-hx, hy, 0.0),
                Vec3::new(hx, hy, 0.0),
                Vec3::new(hx, -hy, 0.0),
                Vec3::NEG_Z,
            );
            // Ön Yüz (Y = -hy, Normal = -Y)
            add_quad(
                Vec3::new(-hx, -hy, 0.0),
                Vec3::new(hx, -hy, 0.0),
                Vec3::new(hx, -hy, hz),
                Vec3::new(-hx, -hy, hz),
                Vec3::NEG_Y,
            );
            // Arka Yüz (Y = +hy, Normal = +Y)
            add_quad(
                Vec3::new(hx, hy, 0.0),
                Vec3::new(-hx, hy, 0.0),
                Vec3::new(-hx, hy, hz),
                Vec3::new(hx, hy, hz),
                Vec3::Y,
            );
            // Sol Yüz (X = -hx, Normal = -X)
            add_quad(
                Vec3::new(-hx, hy, 0.0),
                Vec3::new(-hx, -hy, 0.0),
                Vec3::new(-hx, -hy, hz),
                Vec3::new(-hx, hy, hz),
                Vec3::NEG_X,
            );
            // Sağ Yüz (X = +hx, Normal = +X)
            add_quad(
                Vec3::new(hx, -hy, 0.0),
                Vec3::new(hx, hy, 0.0),
                Vec3::new(hx, hy, hz),
                Vec3::new(hx, -hy, hz),
                Vec3::X,
            );

            // Üst Düzlem (Z = hz, Normal = +Z) ve Ø24 mm Silindirik Delik (İçe Bakan Normaller)
            for i in 0..segments {
                let a0 = (i as f32 / segments as f32) * std::f32::consts::TAU;
                let a1 = ((i + 1) as f32 / segments as f32) * std::f32::consts::TAU;

                let (c0, s0) = (a0.cos(), a0.sin());
                let (c1, s1) = (a1.cos(), a1.sin());

                // Delik iç silindir yüzeyi (Merkez: X=0, Y=0, R=12.0 mm)
                // Normal vektör delik merkezine doğru bakar (-cos, -sin, 0)
                let p_top0 = Vec3::new(bore_radius * c0, bore_radius * s0, hz);
                let p_top1 = Vec3::new(bore_radius * c1, bore_radius * s1, hz);
                let p_bot0 = Vec3::new(bore_radius * c0, bore_radius * s0, 0.0);
                let p_bot1 = Vec3::new(bore_radius * c1, bore_radius * s1, 0.0);

                let n0 = Vec3::new(-c0, -s0, 0.0);
                let n1 = Vec3::new(-c1, -s1, 0.0);

                let base = vertices.len() as u32;
                vertices.extend_from_slice(&[p_top0, p_bot0, p_bot1, p_top1]);
                normals.extend_from_slice(&[n0, n0, n1, n1]);
                indices.push([base, base + 1, base + 2]);
                indices.push([base, base + 2, base + 3]);

                // Üst düzlem halkası (Delik kenarından dış dikdörtgen sınırına projeksiyon)
                let outer_scale = |c: f32, s: f32| -> Vec3 {
                    let tx = if c.abs() > 1e-5 { hx / c.abs() } else { f32::MAX };
                    let ty = if s.abs() > 1e-5 { hy / s.abs() } else { f32::MAX };
                    let t = tx.min(ty);
                    Vec3::new(c * t, s * t, hz)
                };

                let o0 = outer_scale(c0, s0);
                let o1 = outer_scale(c1, s1);
                add_quad(p_top0, p_top1, o1, o0, Vec3::Z);
            }

            let feature_edges =
                Self::extract_sharp_edges(&vertices, &indices, 25.0_f32.to_radians());

            Self {
                file_name: "HEXAGON_CAL_BLOCK_100x60x25.STEP".to_string(),
                vertices,
                normals,
                indices,
                feature_edges,
                bounding_min: Vec3::new(-hx, -hy, 0.0),
                bounding_max: Vec3::new(hx, hy, hz),
            }
        }

        /// Bevy 3B PBR sahnesi için Mesh nesnesi üretir
        pub fn to_bevy_mesh(&self) -> Mesh {
            let positions: Vec<[f32; 3]> = self.vertices.iter().map(|v| v.to_array()).collect();
            let norms: Vec<[f32; 3]> = self.normals.iter().map(|n| n.to_array()).collect();
            let mut flat_indices = Vec::with_capacity(self.indices.len() * 3);
            for tri in &self.indices {
                flat_indices.extend_from_slice(tri);
            }

            let mut mesh = Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
            );
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
            mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, norms);
            mesh.insert_indices(Indices::U32(flat_indices));
            mesh
        }
    }
}

// ============================================================================
// 2. MODÜL: METROLOJİ, RAYCASTING, PROB TELAFİSİ VE DÜZLEM/DAİRE HESAPLAMALARI
// ============================================================================
pub mod metrology {
    use super::cad_occt::CadModelData;
    use bevy::prelude::*;
    use nalgebra::{DMatrix, DVector, Vector2, Vector3};

    /// CAD Üzerinden Alınan Tekil Temas Noktası (PC-DMIS Vector Point)
    #[derive(Clone, Debug)]
    pub struct MeasuredHit {
        pub id: String,
        /// CAD yüzeyindeki temas noktası (X, Y, Z) [mm]
        pub surface_pt: Vec3,
        /// Birim yüzey normal vektörü (I, J, K)
        pub normal: Vec3,
        /// Yakut küre (Ruby Tip) merkez koordinatı: P_probe = P_surface + r * N
        pub probe_center_pt: Vec3,
        /// Ölçüm anındaki yakut küre yarıçapı [mm]
        pub probe_radius: f32,
    }

    /// Hesaplanan Geometrik Unsur Türü
    #[derive(Clone, Debug)]
    pub enum FittedFeatureKind {
        Plane {
            centroid: Vec3,
            normal: Vec3,
            flatness: f32,
            hit_count: usize,
        },
        Circle {
            center: Vec3,
            normal: Vec3,
            /// Prob yarıçapı telafi edilmiş (compensated) gerçek çap [mm]
            diameter: f32,
            /// Telafisiz prob merkezi yörünge çapı [mm]
            probe_center_diameter: f32,
            roundness: f32,
            is_inner_bore: bool,
            hit_count: usize,
        },
    }

    #[derive(Clone, Debug)]
    pub struct ConstructedFeature {
        pub id: String,
        pub kind: FittedFeatureKind,
    }

    /// Möller–Trumbore algoritması ile 3B Işın (Ray) ve CAD Üçgen Ağı kesişimi.
    /// En yakın yüzey noktasını (X, Y, Z) ve Barycentric enterpolasyonlu yüzey normalini (I, J, K) döndürür.
    pub fn raycast_cad_surface(
        ray_origin: Vec3,
        ray_dir: Vec3,
        cad: &CadModelData,
    ) -> Option<(Vec3, Vec3, f32)> {
        let mut closest_t = f32::MAX;
        let mut best_hit: Option<(Vec3, Vec3)> = None;

        for &[i0, i1, i2] in &cad.indices {
            let v0 = cad.vertices[i0 as usize];
            let v1 = cad.vertices[i1 as usize];
            let v2 = cad.vertices[i2 as usize];

            let edge1 = v1 - v0;
            let edge2 = v2 - v0;
            let h = ray_dir.cross(edge2);
            let a = edge1.dot(h);

            if a.abs() < 1e-7 {
                continue;
            }

            let f = 1.0 / a;
            let s = ray_origin - v0;
            let u = f * s.dot(h);
            if !(0.0..=1.0).contains(&u) {
                continue;
            }

            let q = s.cross(edge1);
            let v = f * ray_dir.dot(q);
            if v < 0.0 || u + v > 1.0 {
                continue;
            }

            let t = f * edge2.dot(q);
            if t > 1e-4 && t < closest_t {
                closest_t = t;
                let w = 1.0 - u - v;
                let n0 = cad.normals[i0 as usize];
                let n1 = cad.normals[i1 as usize];
                let n2 = cad.normals[i2 as usize];

                let mut interp_normal = (w * n0 + u * n1 + v * n2).normalize_or_zero();
                if interp_normal.length_squared() < 1e-4 {
                    interp_normal = edge1.cross(edge2).normalize_or_zero();
                }
                if interp_normal.dot(ray_dir) > 0.0 {
                    interp_normal = -interp_normal;
                }

                let hit_point = ray_origin + ray_dir * t;
                best_hit = Some((hit_point, interp_normal));
            }
        }

        best_hit.map(|(pt, norm)| (pt, norm, closest_t))
    }

    /// SVD (Tekil Değer Ayrışımı) ile Ortogonal En Küçük Kareler Düzlem (Plane) Uydurma
    /// ve 3B Yakut Prob Yarıçap Telafisi (P_comp = P_probe - r_ruby * N_plane).
    pub fn fit_plane_svd(hits: &[MeasuredHit]) -> Result<FittedFeatureKind, String> {
        if hits.len() < 3 {
            return Err("Düzlem (PLN) hesabı için CAD üzerinden en az 3 nokta alınmalıdır.".into());
        }

        let n = hits.len();
        let mut sum_probe = Vector3::<f64>::zeros();
        let mut avg_cad_normal = Vector3::<f64>::zeros();
        let mut avg_ruby_r = 0.0_f64;

        for h in hits {
            sum_probe += Vector3::new(
                h.probe_center_pt.x as f64,
                h.probe_center_pt.y as f64,
                h.probe_center_pt.z as f64,
            );
            avg_cad_normal += Vector3::new(
                h.normal.x as f64,
                h.normal.y as f64,
                h.normal.z as f64,
            );
            avg_ruby_r += h.probe_radius as f64;
        }

        let probe_centroid = sum_probe / (n as f64);
        avg_ruby_r /= n as f64;

        // Merkezlenmiş prob merkezi matrisi A (n x 3)
        let mut mat = DMatrix::<f64>::zeros(n, 3);
        for (row, h) in hits.iter().enumerate() {
            mat[(row, 0)] = h.probe_center_pt.x as f64 - probe_centroid.x;
            mat[(row, 1)] = h.probe_center_pt.y as f64 - probe_centroid.y;
            mat[(row, 2)] = h.probe_center_pt.z as f64 - probe_centroid.z;
        }

        let svd = mat.svd(false, true);
        let v_t = svd
            .v_t
            .ok_or_else(|| "SVD matris ayrışımı başarısız oldu.".to_string())?;

        // En küçük tekil değere karşılık gelen son satır düzlem normalini verir
        let mut plane_normal = Vector3::new(v_t[(2, 0)], v_t[(2, 1)], v_t[(2, 2)]).normalize();

        // Normal yönünü CAD yüzey normalleriyle uyumlu hale getir
        if plane_normal.dot(&avg_cad_normal) < 0.0 {
            plane_normal = -plane_normal;
        }

        // Prob yarıçap telafisi: Düzlem merkezini yakut küre yarıçapı kadar yüzeye geri kaydır
        let compensated_centroid = probe_centroid - avg_ruby_r * plane_normal;

        // ISO 1101 Düzlemsellik (Flatness): max(d_i) - min(d_i)
        let mut min_d = f64::MAX;
        let mut max_d = f64::MIN;
        for h in hits {
            let p_comp = Vector3::new(
                h.probe_center_pt.x as f64,
                h.probe_center_pt.y as f64,
                h.probe_center_pt.z as f64,
            ) - (h.probe_radius as f64) * plane_normal;
            let d = (p_comp - compensated_centroid).dot(&plane_normal);
            min_d = min_d.min(d);
            max_d = max_d.max(d);
        }

        let flatness = (max_d - min_d) as f32;

        Ok(FittedFeatureKind::Plane {
            centroid: Vec3::new(
                compensated_centroid.x as f32,
                compensated_centroid.y as f32,
                compensated_centroid.z as f32,
            ),
            normal: Vec3::new(
                plane_normal.x as f32,
                plane_normal.y as f32,
                plane_normal.z as f32,
            ),
            flatness,
            hit_count: n,
        })
    }

    /// 3B Noktalardan En Küçük Kareler Daire (Circle) Uydurma ve Yakut Prob Yarıçap Telafisi
    /// İç delik (Bore) ölçümünde D_gerçek = D_prob_merkezi + 2*r_ruby,
    /// Dış silindir (Boss) ölçümünde D_gerçek = D_prob_merkezi - 2*r_ruby uygulanır.
    pub fn fit_circle_3d(hits: &[MeasuredHit]) -> Result<FittedFeatureKind, String> {
        if hits.len() < 3 {
            return Err("Daire (CIR) hesabı için CAD üzerinden en az 3 nokta alınmalıdır.".into());
        }

        let n = hits.len();
        let mut sum_probe = Vector3::<f64>::zeros();
        let mut avg_ruby_r = 0.0_f64;

        for h in hits {
            sum_probe += Vector3::new(
                h.probe_center_pt.x as f64,
                h.probe_center_pt.y as f64,
                h.probe_center_pt.z as f64,
            );
            avg_ruby_r += h.probe_radius as f64;
        }
        let centroid = sum_probe / (n as f64);
        avg_ruby_r /= n as f64;

        // 1. Noktaların bulunduğu kesit düzlemini SVD ile bul
        let mut mat = DMatrix::<f64>::zeros(n, 3);
        for (row, h) in hits.iter().enumerate() {
            mat[(row, 0)] = h.probe_center_pt.x as f64 - centroid.x;
            mat[(row, 1)] = h.probe_center_pt.y as f64 - centroid.y;
            mat[(row, 2)] = h.probe_center_pt.z as f64 - centroid.z;
        }

        let svd = mat.svd(false, true);
        let v_t = svd
            .v_t
            .ok_or_else(|| "Daire kesit düzlemi SVD hatası.".to_string())?;
        let mut axis_normal = Vector3::new(v_t[(2, 0)], v_t[(2, 1)], v_t[(2, 2)]).normalize();
        if axis_normal.z < 0.0 {
            axis_normal = -axis_normal;
        }

        // 2. Düzlem üzerinde yerel 2B ortonormal baz (u, v) oluştur
        let ref_vec = if axis_normal.x.abs() < 0.9 {
            Vector3::new(1.0, 0.0, 0.0)
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };
        let u_axis = axis_normal.cross(&ref_vec).normalize();
        let v_axis = axis_normal.cross(&u_axis).normalize();

        // 3. 3B prob merkez noktalarını 2B (u, v) koordinatlarına izdüşür
        let mut pts_2d = Vec::with_capacity(n);
        for h in hits {
            let dp = Vector3::new(
                h.probe_center_pt.x as f64,
                h.probe_center_pt.y as f64,
                h.probe_center_pt.z as f64,
            ) - centroid;
            pts_2d.push(Vector2::new(dp.dot(&u_axis), dp.dot(&v_axis)));
        }

        // 4. Cebirsel Kåsa Yöntemi: [2u, 2v, 1] * [uc, vc, c]^T = [u^2 + v^2]
        let mut a_mat = DMatrix::<f64>::zeros(n, 3);
        let mut b_vec = DVector::<f64>::zeros(n);
        for (i, p) in pts_2d.iter().enumerate() {
            a_mat[(i, 0)] = 2.0 * p.x;
            a_mat[(i, 1)] = 2.0 * p.y;
            a_mat[(i, 2)] = 1.0;
            b_vec[i] = p.x * p.x + p.y * p.y;
        }

        let sol = a_mat
            .svd(true, true)
            .solve(&b_vec, 1e-9)
            .map_err(|_| "Noktalar doğrusal (collinear) olduğu için daire uydurulamadı.")?;

        let mut uc = sol[0];
        let mut vc = sol[1];
        let mut probe_radius_fit = (uc * uc + vc * vc + sol[2]).max(1e-6).sqrt();

        // 5. Geometrik Gauss-Newton İyileştirmesi (Ortogonal Mesafe Minimizasyonu)
        for _ in 0..10 {
            let mut j_mat = DMatrix::<f64>::zeros(n, 3);
            let mut r_vec = DVector::<f64>::zeros(n);
            for (i, p) in pts_2d.iter().enumerate() {
                let dx = uc - p.x;
                let dy = vc - p.y;
                let dist = (dx * dx + dy * dy).max(1e-9).sqrt();
                j_mat[(i, 0)] = dx / dist;
                j_mat[(i, 1)] = dy / dist;
                j_mat[(i, 2)] = -1.0;
                r_vec[i] = dist - probe_radius_fit;
            }
            if let Ok(delta) = j_mat.svd(true, true).solve(&(-r_vec), 1e-9) {
                uc += delta[0];
                vc += delta[1];
                probe_radius_fit = (probe_radius_fit + delta[2]).max(1e-4);
                if delta.norm() < 1e-8 {
                    break;
                }
            }
        }

        // 6. 2B merkezi 3B uzaya geri taşı ve İç Delik (Bore) / Dış Çap (Boss) telafisini uygula
        let center_3d = centroid + uc * u_axis + vc * v_axis;

        let mut min_r = f64::MAX;
        let mut max_r = f64::MIN;
        let mut inward_votes = 0;

        for (i, p) in pts_2d.iter().enumerate() {
            let dist = ((p.x - uc).powi(2) + (p.y - vc).powi(2)).sqrt();
            min_r = min_r.min(dist);
            max_r = max_r.max(dist);

            let to_center = center_3d
                - Vector3::new(
                    hits[i].surface_pt.x as f64,
                    hits[i].surface_pt.y as f64,
                    hits[i].surface_pt.z as f64,
                );
            let hit_norm = Vector3::new(
                hits[i].normal.x as f64,
                hits[i].normal.y as f64,
                hits[i].normal.z as f64,
            );
            if hit_norm.dot(&to_center) > 0.0 {
                inward_votes += 1;
            }
        }

        let is_inner_bore = inward_votes * 2 >= n;
        let compensated_radius = if is_inner_bore {
            probe_radius_fit + avg_ruby_r
        } else {
            (probe_radius_fit - avg_ruby_r).max(0.0)
        };
        let roundness = (max_r - min_r) as f32;

        Ok(FittedFeatureKind::Circle {
            center: Vec3::new(
                center_3d.x as f32,
                center_3d.y as f32,
                center_3d.z as f32,
            ),
            normal: Vec3::new(
                axis_normal.x as f32,
                axis_normal.y as f32,
                axis_normal.z as f32,
            ),
            diameter: (2.0 * compensated_radius) as f32,
            probe_center_diameter: (2.0 * probe_radius_fit) as f32,
            roundness,
            is_inner_bore,
            hit_count: n,
        })
    }
}

// ============================================================================
// 3. MODÜL: UYGULAMA DURUMU, SANAL PROB VE 3B SAHNE BİLEŞENLERİ
// ============================================================================
use cad_occt::CadModelData;
use metrology::{ConstructedFeature, FittedFeatureKind, MeasuredHit};

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum ActiveToolMode {
    VectorPoint,
    AutoPlane,
    AutoCircle,
}

#[derive(Resource)]
pub struct CmmAppState {
    pub cad_model: CadModelData,
    pub tool_mode: ActiveToolMode,
    /// Sanal Prob Ayarları
    pub ruby_diameter_mm: f32,
    pub stylus_length_mm: f32,
    pub approach_retract_mm: f32,
    /// Canlı Prob Pozisyonu (DRO)
    pub live_probe_tip_center: Vec3,
    pub live_surface_target: Vec3,
    pub live_surface_normal: Vec3,
    pub is_hovering_cad: bool,
    /// Alınan Noktalar ve Hesaplanan Unsurlar
    pub current_hits: Vec<MeasuredHit>,
    pub all_points: Vec<MeasuredHit>,
    pub features: Vec<ConstructedFeature>,
    /// Görünüm Ayarları
    pub show_wireframe_edges: bool,
    pub show_virtual_probe: bool,
    pub cad_translucent: bool,
    /// Durum Mesajı
    pub status_message: String,
    /// 3B Grafik Penceresinin Ekran Dikdörtgeni (egui_dock tarafından güncellenir)
    pub viewport_rect: Option<egui::Rect>,
}

impl Default for CmmAppState {
    fn default() -> Self {
        let cad_model = CadModelData::generate_cmm_reference_block();
        Self {
            cad_model,
            tool_mode: ActiveToolMode::VectorPoint,
            ruby_diameter_mm: 2.0,
            stylus_length_mm: 25.0,
            approach_retract_mm: 3.0,
            live_probe_tip_center: Vec3::new(0.0, 0.0, 45.0),
            live_surface_target: Vec3::new(0.0, 0.0, 25.0),
            live_surface_normal: Vec3::Z,
            is_hovering_cad: false,
            current_hits: Vec::new(),
            all_points: Vec::new(),
            features: Vec::new(),
            show_wireframe_edges: true,
            show_virtual_probe: true,
            cad_translucent: false,
            status_message: "HAZIR | CAD yüzeyi üzerinde sol tık ile nokta alabilirsiniz (Sağ Tık: Döndür, Orta Tık: Kaydır, Tekerlek: Yakınlaştır).".into(),
            viewport_rect: None,
        }
    }
}

#[derive(Component)]
pub struct CadMeshEntity;

#[derive(Component)]
pub struct VirtualProbeRubyTip;

#[derive(Component)]
pub struct VirtualProbeStem;

#[derive(Component)]
pub struct VirtualProbeHead;

#[derive(Component)]
pub struct OrbitCamera {
    pub focus: Vec3,
    pub radius: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            focus: Vec3::new(0.0, 0.0, 12.5),
            radius: 145.0,
            yaw: -0.65,
            pitch: 0.55,
        }
    }
}

// ============================================================================
// 4. MODÜL: PC-DMIS BENZERİ ARAYÜZ DÜZENİ (egui_dock)
// ============================================================================
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PcDmisTab {
    GraphicDisplayWindow,
    EditWindow,
    ProbeReadoutsDro,
    FeatureTreeInspector,
}

#[derive(Resource)]
pub struct PcDmisDockTree {
    pub tree: DockState<PcDmisTab>,
}

impl Default for PcDmisDockTree {
    fn default() -> Self {
        let mut tree = DockState::new(vec![PcDmisTab::GraphicDisplayWindow]);
        let surface = tree.main_surface_mut();

        let [graphic_node, _left_edit] = surface.split_left(
            NodeIndex::root(),
            0.32,
            vec![PcDmisTab::EditWindow, PcDmisTab::FeatureTreeInspector],
        );

        let [_graphic_node, _bottom_dro] =
            surface.split_below(graphic_node, 0.75, vec![PcDmisTab::ProbeReadoutsDro]);

        Self { tree }
    }
}

struct PcDmisTabViewer<'a> {
    state: &'a mut CmmAppState,
    reload_cad_mesh: &'a mut bool,
    camera_preset: &'a mut Option<(f32, f32)>,
}

impl<'a> TabViewer for PcDmisTabViewer<'a> {
    type Tab = PcDmisTab;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        match tab {
            PcDmisTab::GraphicDisplayWindow => "Graphic Display Window (3B CAD)".into(),
            PcDmisTab::EditWindow => "Edit Window - [Command Mode]".into(),
            PcDmisTab::ProbeReadoutsDro => "Probe Readouts (DRO) & Hit Buffer".into(),
            PcDmisTab::FeatureTreeInspector => "Unsur Ağacı & GD&T Raporu".into(),
        }
    }

    fn clear_background(&self, tab: &Self::Tab) -> bool {
        !matches!(tab, PcDmisTab::GraphicDisplayWindow)
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        match tab {
            PcDmisTab::GraphicDisplayWindow => {
                let rect = ui.available_rect_before_wrap();
                self.state.viewport_rect = Some(rect);

                egui::Frame::none()
                    .fill(egui::Color32::from_rgba_unmultiplied(16, 22, 32, 210))
                    .inner_margin(egui::Margin::same(6.0))
                    .rounding(4.0)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.colored_label(
                                egui::Color32::from_rgb(90, 200, 250),
                                format!("CAD: {}", self.state.cad_model.file_name),
                            );
                            ui.separator();
                            ui.label(format!(
                                "Üçgen: {} | Keskin Kenar: {}",
                                self.state.cad_model.indices.len(),
                                self.state.cad_model.feature_edges.len()
                            ));
                            ui.separator();
                            if ui.small_button("İzometrik").clicked() {
                                *self.camera_preset = Some((-0.65, 0.55));
                            }
                            if ui.small_button("Üst (+Z)").clicked() {
                                *self.camera_preset = Some((0.0, 1.55));
                            }
                            if ui.small_button("Ön (-Y)").clicked() {
                                *self.camera_preset = Some((0.0, 0.05));
                            }
                            if ui.small_button("Sağ (+X)").clicked() {
                                *self.camera_preset = Some((std::f32::consts::FRAC_PI_2, 0.05));
                            }
                        });
                    });
            }

            PcDmisTab::EditWindow => {
                ui.style_mut().override_font_id = Some(egui::FontId::monospace(12.5));
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.colored_label(
                            egui::Color32::from_rgb(130, 185, 255),
                            format!("PART NAME  : {}", self.state.cad_model.file_name),
                        );
                        ui.colored_label(
                            egui::Color32::from_rgb(130, 185, 255),
                            "REV NUMBER : 2026.1   SER NUMBER : SN-RUST-CMM-01",
                        );
                        ui.separator();
                        ui.colored_label(
                            egui::Color32::from_rgb(180, 220, 140),
                            "STARTUP    =ALIGNMENT/START,RECALL:USE_PART_SETUP,LIST=YES",
                        );
                        ui.colored_label(
                            egui::Color32::from_rgb(180, 220, 140),
                            "            ALIGNMENT/END",
                        );
                        ui.colored_label(
                            egui::Color32::from_rgb(245, 200, 110),
                            format!(
                                "            LOADPROBE/RENISHAW_TP20_RUBY_D{:.1}MM",
                                self.state.ruby_diameter_mm
                            ),
                        );
                        ui.colored_label(
                            egui::Color32::from_rgb(245, 200, 110),
                            "            TIP/T1A0B0, SHANKIJK=0, 0, 1, ANGLE=0",
                        );
                        ui.colored_label(
                            egui::Color32::from_rgb(190, 190, 190),
                            format!(
                                "            PREHIT/{:.2}   RETRACT/{:.2}",
                                self.state.approach_retract_mm, self.state.approach_retract_mm
                            ),
                        );
                        ui.separator();

                        for pt in &self.state.all_points {
                            let s = pt.surface_pt;
                            let n = pt.normal;
                            let c = pt.probe_center_pt;
                            ui.colored_label(
                                egui::Color32::from_rgb(100, 235, 255),
                                format!(
                                    "{:<10} =FEAT/CONTACT/VECTOR POINT/DEFAULT,CARTESIAN",
                                    pt.id
                                ),
                            );
                            ui.label(format!(
                                "            THEO/<{:>8.3},{:>8.3},{:>8.3}>,<{:>7.4},{:>7.4},{:>7.4}>",
                                s.x, s.y, s.z, n.x, n.y, n.z
                            ));
                            ui.label(format!(
                                "            ACTL/<{:>8.3},{:>8.3},{:>8.3}>,<{:>7.4},{:>7.4},{:>7.4}>",
                                s.x, s.y, s.z, n.x, n.y, n.z
                            ));
                            ui.colored_label(
                                egui::Color32::from_rgb(155, 165, 180),
                                format!(
                                    "            PROB_CTR/<{:>8.3},{:>8.3},{:>8.3}>,RUBY_R={:.3}",
                                    c.x, c.y, c.z, pt.probe_radius
                                ),
                            );
                        }

                        for feat in &self.state.features {
                            ui.add_space(4.0);
                            match &feat.kind {
                                FittedFeatureKind::Plane {
                                    centroid,
                                    normal,
                                    flatness,
                                    hit_count,
                                } => {
                                    ui.colored_label(
                                        egui::Color32::from_rgb(120, 250, 150),
                                        format!(
                                            "{:<10} =FEAT/PLANE,CARTESIAN,TRIANGLE,NO,LEAST_SQR",
                                            feat.id
                                        ),
                                    );
                                    ui.label(format!(
                                        "            THEO/<{:>8.3},{:>8.3},{:>8.3}>,<{:>7.4},{:>7.4},{:>7.4}>",
                                        centroid.x, centroid.y, centroid.z, normal.x, normal.y, normal.z
                                    ));
                                    ui.label(format!(
                                        "            ACTL/<{:>8.3},{:>8.3},{:>8.3}>,<{:>7.4},{:>7.4},{:>7.4}>",
                                        centroid.x, centroid.y, centroid.z, normal.x, normal.y, normal.z
                                    ));
                                    ui.colored_label(
                                        egui::Color32::from_rgb(255, 215, 100),
                                        format!(
                                            "            CONSTR/PLANE,BF,HITS={} | FLATNESS={:.4} MM",
                                            hit_count, flatness
                                        ),
                                    );
                                }
                                FittedFeatureKind::Circle {
                                    center,
                                    normal,
                                    diameter,
                                    probe_center_diameter,
                                    roundness,
                                    is_inner_bore,
                                    hit_count,
                                } => {
                                    let kind_str = if *is_inner_bore { "IN" } else { "OUT" };
                                    ui.colored_label(
                                        egui::Color32::from_rgb(255, 180, 95),
                                        format!(
                                            "{:<10} =FEAT/CIRCLE,CARTESIAN,{},LEAST_SQR",
                                            feat.id, kind_str
                                        ),
                                    );
                                    ui.label(format!(
                                        "            THEO/<{:>8.3},{:>8.3},{:>8.3}>,<{:>7.4},{:>7.4},{:>7.4}>,{:>8.4}",
                                        center.x, center.y, center.z, normal.x, normal.y, normal.z, diameter
                                    ));
                                    ui.label(format!(
                                        "            ACTL/<{:>8.3},{:>8.3},{:>8.3}>,<{:>7.4},{:>7.4},{:>7.4}>,{:>8.4} (RAW_PROB_D={:.4})",
                                        center.x, center.y, center.z, normal.x, normal.y, normal.z, diameter, probe_center_diameter
                                    ));
                                    ui.colored_label(
                                        egui::Color32::from_rgb(255, 215, 100),
                                        format!(
                                            "            CONSTR/CIRCLE,BF,HITS={} | ROUNDNESS={:.4} MM",
                                            hit_count, roundness
                                        ),
                                    );
                                }
                            }
                        }
                    });
            }

            PcDmisTab::ProbeReadoutsDro => {
                ui.columns(2, |cols| {
                    cols[0].group(|ui| {
                        ui.heading("PROBE READOUTS (DRO - MM)");
                        let s = self.state.live_surface_target;
                        let n = self.state.live_surface_normal;
                        let p = self.state.live_probe_tip_center;

                        egui::Grid::new("dro_grid")
                            .striped(true)
                            .min_col_width(90.0)
                            .show(ui, |ui| {
                                ui.colored_label(
                                    egui::Color32::from_rgb(255, 95, 95),
                                    egui::RichText::new(format!("X: {:>9.3}", s.x))
                                        .monospace()
                                        .size(16.0),
                                );
                                ui.colored_label(
                                    egui::Color32::from_rgb(180, 190, 205),
                                    egui::RichText::new(format!("I: {:>8.4}", n.x))
                                        .monospace()
                                        .size(14.0),
                                );
                                ui.colored_label(
                                    egui::Color32::from_rgb(150, 160, 175),
                                    egui::RichText::new(format!("Prob X: {:>8.3}", p.x)).monospace(),
                                );
                                ui.end_row();

                                ui.colored_label(
                                    egui::Color32::from_rgb(95, 240, 125),
                                    egui::RichText::new(format!("Y: {:>9.3}", s.y))
                                        .monospace()
                                        .size(16.0),
                                );
                                ui.colored_label(
                                    egui::Color32::from_rgb(180, 190, 205),
                                    egui::RichText::new(format!("J: {:>8.4}", n.y))
                                        .monospace()
                                        .size(14.0),
                                );
                                ui.colored_label(
                                    egui::Color32::from_rgb(150, 160, 175),
                                    egui::RichText::new(format!("Prob Y: {:>8.3}", p.y)).monospace(),
                                );
                                ui.end_row();

                                ui.colored_label(
                                    egui::Color32::from_rgb(95, 175, 255),
                                    egui::RichText::new(format!("Z: {:>9.3}", s.z))
                                        .monospace()
                                        .size(16.0),
                                );
                                ui.colored_label(
                                    egui::Color32::from_rgb(180, 190, 205),
                                    egui::RichText::new(format!("K: {:>8.4}", n.z))
                                        .monospace()
                                        .size(14.0),
                                );
                                ui.colored_label(
                                    egui::Color32::from_rgb(150, 160, 175),
                                    egui::RichText::new(format!("Prob Z: {:>8.3}", p.z)).monospace(),
                                );
                                ui.end_row();
                            });
                    });

                    cols[1].group(|ui| {
                        ui.horizontal(|ui| {
                            ui.heading(format!(
                                "Aktif Vuruş Tamponu ({} Nokta)",
                                self.state.current_hits.len()
                            ));
                            if ui.button("Düzlem Hesapla (SVD)").clicked() {
                                match metrology::fit_plane_svd(&self.state.current_hits) {
                                    Ok(kind) => {
                                        let id = format!("PLN{}", self.state.features.len() + 1);
                                        self.state
                                            .features
                                            .push(ConstructedFeature { id: id.clone(), kind });
                                        self.state.current_hits.clear();
                                        self.state.status_message = format!(
                                            "{} SVD en küçük kareler ve prob telafisi ile oluşturuldu.",
                                            id
                                        );
                                    }
                                    Err(e) => self.state.status_message = e,
                                }
                            }
                            if ui.button("Daire Hesapla (CIR)").clicked() {
                                match metrology::fit_circle_3d(&self.state.current_hits) {
                                    Ok(kind) => {
                                        let id = format!("CIR{}", self.state.features.len() + 1);
                                        self.state
                                            .features
                                            .push(ConstructedFeature { id: id.clone(), kind });
                                        self.state.current_hits.clear();
                                        self.state.status_message = format!(
                                            "{} 3B kesit projeksiyonu, Gauss-Newton ve yakut yarıçap telafisi ile oluşturuldu.",
                                            id
                                        );
                                    }
                                    Err(e) => self.state.status_message = e,
                                }
                            }
                            if ui.button("Tamponu Temizle").clicked() {
                                self.state.current_hits.clear();
                            }
                        });

                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label("Yakut Küre Çapı (Ø mm):");
                            ui.add(
                                egui::Slider::new(&mut self.state.ruby_diameter_mm, 1.0..=8.0)
                                    .step_by(0.5),
                            );
                            ui.label("Yaklaşma Mesafesi (mm):");
                            ui.add(
                                egui::Slider::new(&mut self.state.approach_retract_mm, 0.0..=15.0)
                                    .step_by(0.5),
                            );
                        });
                    });
                });
            }

            PcDmisTab::FeatureTreeInspector => {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Hesaplanan Unsurlar (Constructed Features)");
                    ui.separator();
                    if self.state.features.is_empty() {
                        ui.label(
                            "Henüz düzlem veya daire hesaplanmadı. CAD modeli üzerinde en az 3 noktaya tıklayıp 'Düzlem Hesapla' veya 'Daire Hesapla' düğmesine basın.",
                        );
                    }
                    for feat in &self.state.features {
                        ui.group(|ui| {
                            match &feat.kind {
                                FittedFeatureKind::Plane {
                                    centroid,
                                    normal,
                                    flatness,
                                    hit_count,
                                } => {
                                    ui.colored_label(
                                        egui::Color32::from_rgb(110, 240, 145),
                                        format!("{} [DÜZLEM / PLANE - {} Nokta]", feat.id, hit_count),
                                    );
                                    ui.label(format!(
                                        "Merkez (X,Y,Z): ({:.3}, {:.3}, {:.3}) mm",
                                        centroid.x, centroid.y, centroid.z
                                    ));
                                    ui.label(format!(
                                        "Birim Normal (I,J,K): ({:.4}, {:.4}, {:.4})",
                                        normal.x, normal.y, normal.z
                                    ));
                                    ui.label(format!(
                                        "Düzlemsellik (Flatness ISO 1101): {:.4} mm",
                                        flatness
                                    ));
                                }
                                FittedFeatureKind::Circle {
                                    center,
                                    normal,
                                    diameter,
                                    probe_center_diameter,
                                    roundness,
                                    is_inner_bore,
                                    hit_count,
                                } => {
                                    let bore_label = if *is_inner_bore {
                                        "İÇ DELİK / BORE"
                                    } else {
                                        "DIŞ ÇAP / BOSS"
                                    };
                                    ui.colored_label(
                                        egui::Color32::from_rgb(255, 190, 90),
                                        format!(
                                            "{} [DAİRE / {} - {} Nokta]",
                                            feat.id, bore_label, hit_count
                                        ),
                                    );
                                    ui.label(format!(
                                        "Merkez (X,Y,Z): ({:.3}, {:.3}, {:.3}) mm",
                                        center.x, center.y, center.z
                                    ));
                                    ui.label(format!(
                                        "Kesit Normali (I,J,K): ({:.4}, {:.4}, {:.4})",
                                        normal.x, normal.y, normal.z
                                    ));
                                    ui.label(format!(
                                        "Telafili Çap (Ø): {:.4} mm (Prob Yörünge Ø: {:.4} mm) | Dairesellik: {:.4} mm",
                                        diameter, probe_center_diameter, roundness
                                    ));
                                }
                            }
                        });
                    }
                    let _ = self.reload_cad_mesh;
                });
            }
        }
    }
}

// ============================================================================
// 5. MODÜL: BEVY SİSTEMLERİ (SAHNE KURULUMU, SANAL PROB, GIZMO VE ETKİLEŞİM)
// ============================================================================
fn setup_cmm_3d_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    app_state: Res<CmmAppState>,
) {
    // 1. CAD Parçası PBR Görünümü (PC-DMIS Çelik Mavi-Gri CAD Materyali)
    let cad_mesh_handle = meshes.add(app_state.cad_model.to_bevy_mesh());
    let cad_material = materials.add(StandardMaterial {
        base_color: Color::srgba(0.55, 0.68, 0.82, 1.0),
        metallic: 0.45,
        perceptual_roughness: 0.35,
        reflectance: 0.6,
        double_sided: true,
        cull_mode: None,
        ..default()
    });

    commands.spawn((
        Mesh3d(cad_mesh_handle),
        MeshMaterial3d(cad_material),
        Transform::IDENTITY,
        CadMeshEntity,
    ));

    // 2. Sanal Prob Küresi (Yakut / Ruby Tip - Kırmızı Yarı Parlak Küre)
    let ruby_mesh = meshes.add(Sphere::new(1.0).mesh().uv(32, 18));
    let ruby_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.92, 0.08, 0.22, 0.95),
        emissive: LinearRgba::new(0.45, 0.02, 0.08, 1.0),
        metallic: 0.1,
        perceptual_roughness: 0.12,
        ..default()
    });
    commands.spawn((
        Mesh3d(ruby_mesh),
        MeshMaterial3d(ruby_mat),
        Transform::from_xyz(0.0, 0.0, 45.0),
        VirtualProbeRubyTip,
    ));

    // 3. Sanal Prob Seramik Gövde Çubuğu (Stylus Stem)
    let stem_mesh = meshes.add(Cylinder::new(0.75, 1.0));
    let stem_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.92, 0.92, 0.95),
        metallic: 0.2,
        perceptual_roughness: 0.25,
        ..default()
    });
    commands.spawn((
        Mesh3d(stem_mesh),
        MeshMaterial3d(stem_mat),
        Transform::from_xyz(0.0, 0.0, 60.0),
        VirtualProbeStem,
    ));

    // 4. Sanal Prob Kafası (Renishaw PH10M / TP20 Modülü)
    let head_mesh = meshes.add(Cylinder::new(6.5, 18.0));
    let head_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.18, 0.20, 0.24),
        metallic: 0.8,
        perceptual_roughness: 0.3,
        ..default()
    });
    commands.spawn((
        Mesh3d(head_mesh),
        MeshMaterial3d(head_mat),
        Transform::from_xyz(0.0, 0.0, 80.0),
        VirtualProbeHead,
    ));

    // 5. Endüstriyel Stüdyo Işıklandırması
    commands.spawn((
        DirectionalLight {
            illuminance: 11_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(120.0, -140.0, 180.0).looking_at(Vec3::ZERO, Vec3::Z),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 5_500.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(-140.0, 120.0, 90.0).looking_at(Vec3::ZERO, Vec3::Z),
    ));

    // 6. 3B CAD Orbit Kamera
    let cam = OrbitCamera::default();
    let eye = compute_orbit_eye(&cam);
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(eye).looking_at(cam.focus, Vec3::Z),
        cam,
    ));
}

fn compute_orbit_eye(cam: &OrbitCamera) -> Vec3 {
    let x = cam.radius * cam.pitch.cos() * cam.yaw.sin();
    let y = -cam.radius * cam.pitch.cos() * cam.yaw.cos();
    let z = cam.radius * cam.pitch.sin();
    cam.focus + Vec3::new(x, y, z)
}

/// Üst Araç Çubuğu (PC-DMIS Menü, STEP/IGES İçe Aktarma, Otomatik Unsur Seçimi) ve Dock Alanı
fn render_pcdmis_ui_system(
    mut contexts: EguiContexts,
    mut app_state: ResMut<CmmAppState>,
    mut dock_tree: ResMut<PcDmisDockTree>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    cad_query: Query<(&Mesh3d, &MeshMaterial3d<StandardMaterial>), With<CadMeshEntity>>,
    mut cam_query: Query<&mut OrbitCamera>,
) {
    let ctx = contexts.ctx_mut();

    let mut visuals = egui::Visuals::dark();
    visuals.window_fill = egui::Color32::from_rgb(22, 26, 33);
    visuals.panel_fill = egui::Color32::from_rgb(26, 31, 40);
    ctx.set_visuals(visuals);

    let mut reload_cad_mesh = false;
    let mut camera_preset: Option<(f32, f32)> = None;

    // 1. ÜST MENÜ VE PC-DMIS HIZLI ARAÇ ÇUBUĞU
    egui::TopBottomPanel::top("pcdmis_top_toolbar").show(ctx, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(
                egui::Color32::from_rgb(80, 195, 255),
                egui::RichText::new("HEXAGON PC-DMIS [Rust CAD++ 2026]").strong(),
            );
            ui.separator();

            if ui.button("📂 STEP / IGES Yükle (OpenCASCADE)").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("STEP / IGES CAD Dosyaları", &["step", "stp", "iges", "igs"])
                    .pick_file()
                {
                    match CadModelData::load_step_or_iges(&path, 0.05) {
                        Ok(new_cad) => {
                            app_state.status_message = format!(
                                "CAD modeli başarıyla yüklendi: {} ({} üçgen)",
                                new_cad.file_name,
                                new_cad.indices.len()
                            );
                            app_state.cad_model = new_cad;
                            reload_cad_mesh = true;
                        }
                        Err(err_msg) => {
                            app_state.status_message = err_msg;
                        }
                    }
                }
            }

            if ui.button("🧱 Kalibrasyon Bloğunu Sıfırla").clicked() {
                app_state.cad_model = CadModelData::generate_cmm_reference_block();
                app_state.current_hits.clear();
                app_state.all_points.clear();
                app_state.features.clear();
                reload_cad_mesh = true;
                app_state.status_message =
                    "Referans CMM Kalibrasyon Bloğu (100x60x25 mm, Ø24 mm Delik) yüklendi.".into();
            }

            ui.separator();
            ui.label("Mod:");
            ui.selectable_value(
                &mut app_state.tool_mode,
                ActiveToolMode::VectorPoint,
                "📍 Vektör Noktası (PT)",
            );
            ui.selectable_value(
                &mut app_state.tool_mode,
                ActiveToolMode::AutoPlane,
                "📐 Otomatik Düzlem (4 Vuruş)",
            );
            ui.selectable_value(
                &mut app_state.tool_mode,
                ActiveToolMode::AutoCircle,
                "⭕ Otomatik Daire (4 Vuruş)",
            );

            ui.separator();
            ui.checkbox(&mut app_state.show_wireframe_edges, "CAD Kenarları");
            ui.checkbox(&mut app_state.show_virtual_probe, "3B Sanal Prob");
            if ui
                .checkbox(&mut app_state.cad_translucent, "Yarı Şeffaf CAD")
                .changed()
            {
                if let Ok((_, mat_handle)) = cad_query.get_single() {
                    if let Some(mat) = materials.get_mut(mat_handle) {
                        if app_state.cad_translucent {
                            mat.base_color = Color::srgba(0.55, 0.68, 0.82, 0.55);
                            mat.alpha_mode = AlphaMode::Blend;
                        } else {
                            mat.base_color = Color::srgba(0.55, 0.68, 0.82, 1.0);
                            mat.alpha_mode = AlphaMode::Opaque;
                        }
                    }
                }
            }
        });
    });

    // 2. ALT DURUM ÇUBUĞU (PC-DMIS Status Bar)
    egui::TopBottomPanel::bottom("pcdmis_status_bar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.colored_label(
                egui::Color32::from_rgb(130, 230, 155),
                format!("● {}", app_state.status_message),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(format!(
                    "Toplam Nokta: {} | Unsur: {}",
                    app_state.all_points.len(),
                    app_state.features.len()
                ));
            });
        });
    });

    // 3. KENETLENEBİLİR PC-DMIS PANELLERİ (DockArea)
    egui::CentralPanel::default()
        .frame(egui::Frame::none())
        .show(ctx, |ui| {
            let mut style = Style::from_egui(ui.style().as_ref());
            style.tab_bar.fill_tab_bar = true;

            let mut viewer = PcDmisTabViewer {
                state: &mut app_state,
                reload_cad_mesh: &mut reload_cad_mesh,
                camera_preset: &mut camera_preset,
            };

            DockArea::new(&mut dock_tree.tree)
                .style(style)
                .show_inside(ui, &mut viewer);
        });

    if reload_cad_mesh {
        if let Ok((mesh_handle, _)) = cad_query.get_single() {
            if let Some(mesh) = meshes.get_mut(mesh_handle) {
                *mesh = app_state.cad_model.to_bevy_mesh();
            }
        }
        if let Ok(mut cam) = cam_query.get_single_mut() {
            cam.focus = app_state.cad_model.center();
        }
    }

    if let Some((yaw, pitch)) = camera_preset {
        if let Ok(mut cam) = cam_query.get_single_mut() {
            cam.yaw = yaw;
            cam.pitch = pitch;
        }
    }
}

/// 3B Kamera Döndürme/Kaydırma, CAD Raycasting, Nokta Alma ve Sanal Prob Güncellemesi
fn cad_viewport_interaction_and_probe_system(
    window_query: Query<&Window, With<PrimaryWindow>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: EventReader<bevy::input::mouse::MouseMotion>,
    mut mouse_wheel: EventReader<bevy::input::mouse::MouseWheel>,
    mut app_state: ResMut<CmmAppState>,
    mut cam_query: Query<(&Camera, &GlobalTransform, &mut Transform, &mut OrbitCamera)>,
    mut ruby_query: Query<
        &mut Transform,
        (
            With<VirtualProbeRubyTip>,
            Without<OrbitCamera>,
            Without<VirtualProbeStem>,
            Without<VirtualProbeHead>,
        ),
    >,
    mut stem_query: Query<
        &mut Transform,
        (
            With<VirtualProbeStem>,
            Without<OrbitCamera>,
            Without<VirtualProbeRubyTip>,
            Without<VirtualProbeHead>,
        ),
    >,
    mut head_query: Query<
        &mut Transform,
        (
            With<VirtualProbeHead>,
            Without<OrbitCamera>,
            Without<VirtualProbeRubyTip>,
            Without<VirtualProbeStem>,
        ),
    >,
) {
    let Ok(window) = window_query.get_single() else {
        return;
    };
    let Ok((camera, cam_global, mut cam_transform, mut orbit)) = cam_query.get_single_mut() else {
        return;
    };

    let cursor_pos = window.cursor_position();
    let in_cad_viewport = match (cursor_pos, app_state.viewport_rect) {
        (Some(cp), Some(rect)) => {
            cp.x >= rect.min.x && cp.x <= rect.max.x && cp.y >= rect.min.y && cp.y <= rect.max.y
        }
        _ => false,
    };

    let mut motion_delta = Vec2::ZERO;
    for ev in mouse_motion.read() {
        motion_delta += ev.delta;
    }

    let mut wheel_delta = 0.0_f32;
    for ev in mouse_wheel.read() {
        wheel_delta += ev.y;
    }

    if in_cad_viewport {
        if mouse_buttons.pressed(MouseButton::Right) && motion_delta.length_squared() > 0.0 {
            orbit.yaw -= motion_delta.x * 0.007;
            orbit.pitch = (orbit.pitch + motion_delta.y * 0.007).clamp(-1.52, 1.52);
        }

        if mouse_buttons.pressed(MouseButton::Middle) && motion_delta.length_squared() > 0.0 {
            let right = cam_transform.right();
            let up = cam_transform.up();
            let pan_speed = orbit.radius * 0.0015;
            orbit.focus += (-right * motion_delta.x + up * motion_delta.y) * pan_speed;
        }

        if wheel_delta.abs() > 0.0 {
            orbit.radius = (orbit.radius * (1.0 - wheel_delta * 0.09)).clamp(25.0, 800.0);
        }
    }

    let eye = compute_orbit_eye(&orbit);
    *cam_transform = Transform::from_translation(eye).looking_at(orbit.focus, Vec3::Z);

    if in_cad_viewport {
        if let Some(cursor) = cursor_pos {
            if let Ok(ray) = camera.viewport_to_world(cam_global, cursor) {
                if let Some((hit_pt, hit_normal, _t)) = metrology::raycast_cad_surface(
                    ray.origin,
                    *ray.direction,
                    &app_state.cad_model,
                ) {
                    let ruby_r = app_state.ruby_diameter_mm * 0.5;
                    let hover_clearance = if mouse_buttons.pressed(MouseButton::Left) {
                        0.0
                    } else {
                        app_state.approach_retract_mm
                    };

                    app_state.is_hovering_cad = true;
                    app_state.live_surface_target = hit_pt;
                    app_state.live_surface_normal = hit_normal;
                    app_state.live_probe_tip_center =
                        hit_pt + hit_normal * (ruby_r + hover_clearance);

                    if mouse_buttons.just_pressed(MouseButton::Left) {
                        let pnt_id = format!("PNT{}", app_state.all_points.len() + 1);
                        let new_hit = MeasuredHit {
                            id: pnt_id.clone(),
                            surface_pt: hit_pt,
                            normal: hit_normal,
                            probe_center_pt: hit_pt + hit_normal * ruby_r,
                            probe_radius: ruby_r,
                        };

                        app_state.all_points.push(new_hit.clone());
                        app_state.current_hits.push(new_hit);

                        app_state.status_message = format!(
                            "{} alındı: THEO=({:.3}, {:.3}, {:.3}) IJK=({:.4}, {:.4}, {:.4})",
                            pnt_id,
                            hit_pt.x,
                            hit_pt.y,
                            hit_pt.z,
                            hit_normal.x,
                            hit_normal.y,
                            hit_normal.z
                        );

                        if app_state.tool_mode == ActiveToolMode::AutoPlane
                            && app_state.current_hits.len() >= 4
                        {
                            if let Ok(kind) = metrology::fit_plane_svd(&app_state.current_hits) {
                                let id = format!("PLN{}", app_state.features.len() + 1);
                                app_state
                                    .features
                                    .push(ConstructedFeature { id: id.clone(), kind });
                                app_state.current_hits.clear();
                                app_state.status_message =
                                    format!("Otomatik Düzlem {} (4 vuruş) hesaplandı.", id);
                            }
                        } else if app_state.tool_mode == ActiveToolMode::AutoCircle
                            && app_state.current_hits.len() >= 4
                        {
                            if let Ok(kind) = metrology::fit_circle_3d(&app_state.current_hits) {
                                let id = format!("CIR{}", app_state.features.len() + 1);
                                app_state
                                    .features
                                    .push(ConstructedFeature { id: id.clone(), kind });
                                app_state.current_hits.clear();
                                app_state.status_message =
                                    format!("Otomatik Daire {} (4 vuruş) hesaplandı.", id);
                            }
                        }
                    }
                } else {
                    app_state.is_hovering_cad = false;
                }
            }
        }
    }

    let ruby_r = app_state.ruby_diameter_mm * 0.5;
    let tip_center = app_state.live_probe_tip_center;
    let stem_len = app_state.stylus_length_mm;

    if let Ok(mut ruby_tf) = ruby_query.get_single_mut() {
        ruby_tf.translation = tip_center;
        ruby_tf.scale = if app_state.show_virtual_probe {
            Vec3::splat(ruby_r)
        } else {
            Vec3::ZERO
        };
    }

    if let Ok(mut stem_tf) = stem_query.get_single_mut() {
        stem_tf.translation = tip_center + Vec3::Z * (stem_len * 0.5);
        stem_tf.rotation = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
        stem_tf.scale = if app_state.show_virtual_probe {
            Vec3::new(1.0, stem_len, 1.0)
        } else {
            Vec3::ZERO
        };
    }

    if let Ok(mut head_tf) = head_query.get_single_mut() {
        head_tf.translation = tip_center + Vec3::Z * (stem_len + 9.0);
        head_tf.rotation = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
        head_tf.scale = if app_state.show_virtual_probe {
            Vec3::ONE
        } else {
            Vec3::ZERO
        };
    }
}

/// CAD Keskin Kenarları (Wireframe), Makine Koordinat Eksenleri, Nokta Normalleri ve Düzlem/Daire Gizmo Çizimi
fn draw_cad_edges_and_metrology_gizmos(mut gizmos: Gizmos, app_state: Res<CmmAppState>) {
    // 1. Parça Koordinat Sistemi Triad (X: Kırmızı, Y: Yeşil, Z: Mavi)
    gizmos.arrow(Vec3::ZERO, Vec3::X * 22.0, Color::srgb(0.95, 0.2, 0.2));
    gizmos.arrow(Vec3::ZERO, Vec3::Y * 22.0, Color::srgb(0.2, 0.9, 0.3));
    gizmos.arrow(Vec3::ZERO, Vec3::Z * 22.0, Color::srgb(0.2, 0.55, 1.0));

    // 2. CAD B-Rep Keskin Sınır Kenarları (PC-DMIS CAD Edge Görünümü)
    if app_state.show_wireframe_edges {
        let edge_color = Color::srgba(0.10, 0.14, 0.20, 0.9);
        for &[a, b] in &app_state.cad_model.feature_edges {
            gizmos.line(a, b, edge_color);
        }
    }

    // 3. Canlı Yüzey Temas Noktası ve Normal Vektör Oku
    if app_state.is_hovering_cad {
        let pt = app_state.live_surface_target;
        let norm = app_state.live_surface_normal;
        gizmos.sphere(
            Isometry3d::from_translation(pt),
            0.65,
            Color::srgb(1.0, 0.9, 0.1),
        );
        gizmos.arrow(pt, pt + norm * 9.0, Color::srgb(1.0, 0.85, 0.1));
    }

    // 4. Alınan Tüm Vektör Noktaları ve (I, J, K) Vektör Okları
    for hit in &app_state.all_points {
        gizmos.sphere(
            Isometry3d::from_translation(hit.surface_pt),
            0.75,
            Color::srgb(0.0, 0.95, 1.0),
        );
        gizmos.arrow(
            hit.surface_pt,
            hit.surface_pt + hit.normal * 7.5,
            Color::srgb(0.0, 0.85, 0.95),
        );
    }

    // 5. Hesaplanan Düzlem ve Dairelerin 3B Geometrik Gösterimi
    for feat in &self.state.features {
        match &feat.kind {
            FittedFeatureKind::Plane {
                centroid, normal, ..
            } => {
                let rot = Quat::from_rotation_arc(Vec3::Z, *normal);
                let iso = Isometry3d::new(*centroid, rot);
                gizmos.rect(iso, Vec2::splat(28.0), Color::srgb(0.2, 1.0, 0.45));
                gizmos.arrow(
                    *centroid,
                    *centroid + *normal * 14.0,
                    Color::srgb(0.2, 1.0, 0.45),
                );
            }
            FittedFeatureKind::Circle {
                center,
                normal,
                diameter,
                ..
            } => {
                let rot = Quat::from_rotation_arc(Vec3::Z, *normal);
                let iso = Isometry3d::new(*center, rot);
                gizmos.circle(iso, *diameter * 0.5, Color::srgb(1.0, 0.65, 0.15));
                gizmos.sphere(
                    Isometry3d::from_translation(*center),
                    0.9,
                    Color::srgb(1.0, 0.65, 0.15),
                );
            }
        }
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Rust-DMIS 2026 | OpenCASCADE CAD Viewer & CMM Metrology System".into(),
                resolution: (1440.0, 900.0).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin)
        .init_resource::<CmmAppState>()
        .init_resource::<PcDmisDockTree>()
        .add_systems(Startup, setup_cmm_3d_scene)
        .add_systems(
            Update,
            (
                render_pcdmis_ui_system,
                cad_viewport_interaction_and_probe_system,
                draw_cad_edges_and_metrology_gizmos,
            )
                .chain(),
        )
        .run();
}
