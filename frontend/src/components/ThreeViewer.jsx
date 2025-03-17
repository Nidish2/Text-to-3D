import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";

function ThreeViewer({ gltfUrl }) {
  const mountRef = useRef(null);

  const sceneObjects = React.useMemo(() => {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    const controls = new OrbitControls(camera, renderer.domElement);
    return { scene, camera, renderer, controls };
  }, []);

  const { scene, camera, renderer, controls } = sceneObjects;

  useEffect(() => {
    const mount = mountRef.current;
    const width = mount.clientWidth;
    const height = mount.clientHeight;

    renderer.setSize(width, height);
    renderer.setClearColor(0xe5e7eb, 1);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    mount.appendChild(renderer.domElement);

    camera.position.set(0, 5, 10);
    camera.lookAt(0, 0, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 0, 0);
    controls.update();

    // Improved lighting
    const directionalLight1 = new THREE.DirectionalLight(0xffffff, 1.5);
    directionalLight1.position.set(5, 5, 5);
    scene.add(directionalLight1);
    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 1.0);
    directionalLight2.position.set(-5, 5, -5);
    scene.add(directionalLight2);
    scene.add(new THREE.AmbientLight(0xffffff, 0.8));

    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      const width = mount.clientWidth;
      const height = mount.clientHeight;
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      mount.removeChild(renderer.domElement);
    };
  }, [camera, controls, renderer, scene]);

  useEffect(() => {
    if (gltfUrl) {
      console.log("Loading GLTF from:", gltfUrl);
      const loader = new GLTFLoader();
      loader.load(
        gltfUrl,
        (gltf) => {
          console.log("GLTF loaded successfully");
          // Clear previous models, keep lights
          scene.children = scene.children.filter((child) => child.isLight);
          const model = gltf.scene;
          scene.add(model);

          // Default material if none exists
          model.traverse((child) => {
            if (child.isMesh && !child.material) {
              child.material = new THREE.MeshStandardMaterial({
                color: 0xff8000, // Orange fallback
                metalness: 0.5,
                roughness: 0.5,
              });
            }
          });

          const box = new THREE.Box3().setFromObject(model);
          const center = box.getCenter(new THREE.Vector3());
          const size = box.getSize(new THREE.Vector3()).length();
          model.position.sub(center);
          if (size > 10) {
            const scale = 10 / size;
            model.scale.set(scale, scale, scale);
          }

          camera.position.set(0, size / 2, size * 2);
          camera.lookAt(0, 0, 0);
          controls.target.set(0, 0, 0);
          controls.update();

          console.log("Model bounding box:", box.min, box.max);
          console.log("Scene children:", scene.children);
        },
        undefined,
        (error) => console.error("Error loading GLTF:", error)
      );
    }
  }, [gltfUrl, scene, camera, controls]);

  return <div ref={mountRef} className="w-full h-96 bg-gray-200 rounded-lg" />;
}

export default ThreeViewer;
