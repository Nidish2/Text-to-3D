import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";

function ThreeViewer({ gltfUrl }) {
  const mountRef = useRef(null);

  // Memoize scene objects to prevent recreation on every render
  const sceneObjects = React.useMemo(() => {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      75, // Field of view
      window.innerWidth / window.innerHeight, // Aspect ratio (will be updated later)
      0.1, // Near plane
      1000 // Far plane
    );
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    const controls = new OrbitControls(camera, renderer.domElement);
    return { scene, camera, renderer, controls };
  }, []);

  const { scene, camera, renderer, controls } = sceneObjects;

  // Setup scene, renderer, and controls
  useEffect(() => {
    const mount = mountRef.current;
    const width = mount.clientWidth;
    const height = mount.clientHeight;

    // Configure renderer
    renderer.setSize(width, height);
    renderer.setClearColor(0xe5e7eb, 1); // Match bg-gray-200 to debug rendering
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    mount.appendChild(renderer.domElement);

    // Initial camera setup
    camera.position.set(0, 5, 10);
    camera.lookAt(0, 0, 0); // Ensure camera looks at the origin
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 0, 0); // Controls target the scene center
    controls.update();

    // Add lighting
    const directionalLight = new THREE.DirectionalLight(0xffffff, 2);
    directionalLight.position.set(5, 5, 5);
    scene.add(directionalLight);
    scene.add(new THREE.AmbientLight(0xffffff, 0.5));

    // Add test cube for debugging
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
    const cube = new THREE.Mesh(geometry, material);
    cube.position.set(0, 0, 0); // Explicitly at origin
    scene.add(cube);

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Handle resize
    const handleResize = () => {
      const width = mount.clientWidth;
      const height = mount.clientHeight;
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    window.addEventListener("resize", handleResize);

    // Cleanup
    return () => {
      window.removeEventListener("resize", handleResize);
      mount.removeChild(renderer.domElement);
    };
  }, [camera, controls, renderer, scene]);

  // Load and position the GLB model
  useEffect(() => {
    if (gltfUrl) {
      console.log("Loading GLTF from:", gltfUrl);
      const loader = new GLTFLoader();
      loader.load(
        gltfUrl,
        (gltf) => {
          console.log("GLTF loaded successfully");
          // Temporarily disable cleanup to keep test cube
          // while (scene.children.length > 2) {
          //   scene.remove(scene.children[2]);
          // }
          scene.add(gltf.scene);

          // Center the model
          const box = new THREE.Box3().setFromObject(gltf.scene);
          const center = box.getCenter(new THREE.Vector3());
          const size = box.getSize(new THREE.Vector3()).length();
          gltf.scene.position.sub(center);

          // Scale if too large
          if (size > 10) {
            const scale = 10 / size;
            gltf.scene.scale.set(scale, scale, scale);
          }

          // Adjust camera
          camera.position.set(0, size / 2, size * 2);
          camera.lookAt(0, 0, 0); // Ensure camera looks at center
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
