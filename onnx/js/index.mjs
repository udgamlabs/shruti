// Shruti: offline speech-to-text, one language and English in one small model, in the browser (onnxruntime-web) or Node.
//
//   import * as ort from "onnxruntime-web";
//   import { createShruti } from "shruti-onnx";
//   const asr = await createShruti({ ort, language: "urdu" });   // a plain word or a code ("ur"); see LANGUAGES
//   const { text } = await asr.transcribe(samples16k);            // Float32Array, 16 kHz mono
//
//   createShruti({ ort, language: "ur", variant: "int8" })        // low memory; the default is the 4-bit build
//   createShruti({ ort, baseUrl: "/models/ur/onnx/" })            // your own copy: a URL, or a folder path in Node
//   asr.transcribe(samples16k, { language: "en" })                // null (the default) detects: the model's language or English
//
// Decoding is 5-beam search with length-normalised scores, the same search as the Python package (shruti_onnx).

export const RATE = 16000;
const LOOPS_DEFAULT = "off";
const REPO = "https://huggingface.co/udgamlabs/Shruti-ONNX/resolve/main/";
const VARIANTS = { "4bit": "onnx", int8: "onnx-int8" };

// sha256 of each released model file on the Hub, keyed by [language code][variant folder][filename]; checked against the
// fetched bytes before they reach ort.InferenceSession.create (a compromised baseUrl, a compromised/typosquatted mirror, or a
// MITM'd plain-HTTP fetch can otherwise substitute a model that still "works"). frontend.onnx/encoder.onnx/decoder.onnx are
// Git LFS blobs on the Hub, so their hash here is the LFS "oid" the Hub already computed on upload (the same integrity root
// onnx/python's huggingface_hub download already relies on, per PR #1's fix for the old examples/ tree); shruti.json is a
// plain git blob (not LFS), so its hash here is sha256 of the file's own bytes, computed by downloading the (94 KB) file once.
// Captured 2026-11-10 from https://huggingface.co/api/models/udgamlabs/Shruti-ONNX/tree/main/<code>/<onnx|onnx-int8> (and the
// file itself, for shruti.json). A custom baseUrl (your own copy, or SHRUTI_MODEL_DIR / the samples' ?local flag) has no entry
// here and loads unverified; createShruti() warns once when that happens. TODO(maintainer): this table was built by an outside
// contributor from the Hub's current metadata, not exported alongside the models themselves — please confirm these match what
// you intend to ship, and re-capture them (ideally from set_release.py, so this can't drift from a real release) whenever the
// Hub repo is updated.
const HASHES = {
  as: { onnx: { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "b0e0b42b298261e9397e1967fd386b4b7b1b7664c2ffaf9d3ca2aa27b61e6f76", "decoder.onnx": "11c828de051f68e684b2743d2b1b0d38b4ba558dcc1049ebdbcde1007ebae6c4", "shruti.json": "1892a0497e147e3a0274349472bbdbd9f9a16ec837c04a4152d4ec09b516b5cf" }, "onnx-int8": { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "3f88118dc4982a7bfdff279e676bf3c71e3e110483a28bcabb6a02e03bf11757", "decoder.onnx": "9db2041a7c33809a776d6c2416e22d67fa9ad9c476f389dd3b9e0fdfc6b8a46b", "shruti.json": "d6b8edf027dee2bcb0eb29b9ece5d271283d06d7d41446312fbefe87ea317346" } },
  bn: { onnx: { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "886aeb5d241fcbf1e9d1dbbe0f84873f36439b2f9f3c47982a9c5221f4ebc4b8", "decoder.onnx": "69d7beef52437bfb25f377d3e031951975b94bff070b2d8cc15af7ec85743767", "shruti.json": "035c8b2775269ba9589296376fb5a2ba3ed3d2a74648658459bee9c35714e522" }, "onnx-int8": { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "3a4be6b110addddf4c42354ca9c2116c4382cde3ea472150085762753120430e", "decoder.onnx": "d64769a5597232c2e0f3750f9b65faad3c9c5ead38446f63fd1507fd3f0202f1", "shruti.json": "c0c6e45ff1252430b001511dc48dd7743ebb7ba1705693357fa8fd5c0bcd1fff" } },
  brx: { onnx: { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "c3dc8d5320916582cf6777d7395ec2ddba278a415397917c262811795bd6e9e9", "decoder.onnx": "e95d86e89bb6988e50bef45f54897ba854a663e4729d3d472d785b53c2608f95", "shruti.json": "754d0366dcc76ad8cf644b8028e32ac7549db25f6708ad4be80ea1ad2317cb58" }, "onnx-int8": { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "c573c012494008411fdcbcf2564cb5bf8b2c7938f24e324d80c578d84dbbf74c", "decoder.onnx": "cf80a7052c7a0abe7b0f81f8d166face34412db9eecce6b083ea256e01db9abe", "shruti.json": "639efd84e56cb611e40a99e5a73841cbd47a35f768fb20a1d9517aea2f2ce2ad" } },
  doi: { onnx: { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "6087c226677f316ed0a449d2b282715900b59ee8bafd0cb95b0685eb27b317fb", "decoder.onnx": "512710c0fb8574dd78c0ee07c35494248d940dcd277707aebeb072838f0fc96b", "shruti.json": "e678f001ac87d242441d1e8084a3b1fcc3665ac3f7f957637fa42ded54d5c207" }, "onnx-int8": { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "790b7ef56e5307fbaebc46721b272bc3982e6c9a6d5ff757e6bce1d7264615c1", "decoder.onnx": "61a545a9fc72147decaa7219afc94d76ad8cd45ad67a32ffb7b595a2c3f8ae51", "shruti.json": "af5b734558e34ad22ef01265e8c3c676f75f7fce28d63f8f2e303a7c6abcdaf5" } },
  gu: { onnx: { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "f1b55fbaced01a41063ba27e8f5f1a875e29fae3447993f12d18fb828ee534e3", "decoder.onnx": "c9f2c076427f3b87ca616283fcec7bace5129a6dc882d1eabef0504c8c2e87e3", "shruti.json": "f31cc3804a427c0af7ad87abfd72c6cde4b9cbf2d8ac72b9f72b82ce140876b9" }, "onnx-int8": { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "3ea590963ccc4f2f6248596eac90163f609d3c7977000981852d9dae594a5bd6", "decoder.onnx": "ec79ac85899e949dadf11b929b3b0466804e03a193348e5dab67706dbbe37346", "shruti.json": "73cbc335c50be19f0a9de2c03a5f61ef2566c5da3e703ad587e218e274418461" } },
  hi: { onnx: { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "8935017652b2fec8932d9eb220a4ae1d5287b8b05904820eadabdc64c0de63d0", "decoder.onnx": "d6ad12693b4289fe5d8f23af5c87dffae9597aa5a585fffaea82fe54a3f27cc3", "shruti.json": "a97f9775bce34d77c23536c2b529eb2f508e950a5948b908bb316708d857ecc0" }, "onnx-int8": { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "b26a33cedf74ce11f13ad3854f3c0a34e7272c4e2dfd0d5cd2818185b029b996", "decoder.onnx": "777870e340af2c5a9513f8bef14ecf0c2d57ba7dbfe8682a99674ec58dbd6f70", "shruti.json": "238ccedf874fc74c9167db85890463a77a4d383bbf03dffb1ba10c7ff193a249" } },
  kn: { onnx: { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "048bf3a49afb8d349df89acf4489d8132952da4a198155dde29810a64fced44d", "decoder.onnx": "b5941b81e4c3c3e0afc649c75210532c4655ade89a7e9ed85c9068f020c1e631", "shruti.json": "abbed2497f1990efcd547a789de7523d8a903ba59c7e9e05bbbf91fa74bc7d8a" }, "onnx-int8": { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "e117c8155632c8fa7b2f9ab15bae6b0302e295c6c89aad04f6250c0a71d40a95", "decoder.onnx": "92626403743146fae3b9f5952b164eadc92c2cf129a1734bcb6d61595840cc50", "shruti.json": "6f9a1ab04118efc036f5466f73970998c2f5f4bc41711774d3aeebeb6cc08260" } },
  kok: { onnx: { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "429da0d4b75e092b508b516e652fb223870e59a555218bd31fc7fdfac813dc6f", "decoder.onnx": "f762a325e69d5c8ca4f4e2278eba8387e3e1f2199594619e15a4adc791748f21", "shruti.json": "0f0f63ac4b686343eab3bee6c3d6fb9b02ea4df773979434889782bfeb717c75" }, "onnx-int8": { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "e7d12dbd7d74d80723e72018f9afe754e5251979cc58acd46ba43a1961b1bdd2", "decoder.onnx": "0e71051f7a9f26bcaa972f2509ce3ec3e48072be4d908907000d91fd7766a9e5", "shruti.json": "cc376624405664cd07b57d8f3460e9f788f3a0e11a166a8d13216d055be7dcd3" } },
  ks: { onnx: { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "7914742f2a11a178abc1c2dff833c7136511f63507c9b2c7eca62c5253798992", "decoder.onnx": "4ef3d802cb439ddc9952c43ba57729b14987e318c68ee7cff5ea971910ae09e6", "shruti.json": "50893ab7d62d2391d0004711027393ecb0c466c0e3d74c6bc3a81c4df8ae6922" }, "onnx-int8": { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "8e9a41a6373d23a91c7d38b4b19ff8d12665bbb5437e6950e2e0d7a296652a49", "decoder.onnx": "7099488ead27ffac51bb23ab62a72f772c827667df26933bebfce21cd49a2bb1", "shruti.json": "f024377a1ef78578dddd48f9d0c5a37edc8ab0e9544f1cc232524842ba7931dd" } },
  mai: { onnx: { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "b73fc335272a5990a3b75b026e5120dcad3cd2e079230dd4ba3b4b354aa29a55", "decoder.onnx": "fb4c5f2cb0fa3bb22b177bf115bc500d7a014ad83d971f38fc62b761ec2a1e25", "shruti.json": "a1e133ede3bc54282b41c169b734f00e87c6a4b391c1ae9f4ff0472e34619a64" }, "onnx-int8": { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "fa4641c528ab641457b84de4ef16a882b73b673e4413c7a1e4a70faf7e2eec17", "decoder.onnx": "d5c4b4b4c84c86b2333495e7f8b348c6ae6b7f34ddf87fd73546c34f649f52ec", "shruti.json": "7b17f47d60e30855428ccf9cc4043bb07eeeb23eed27fe325f5b303f6993fb3d" } },
  ml: { onnx: { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "865135b515b9c20b3e51845fdfc41d9c2327a1cc86c82a5730310522c76a4bd4", "decoder.onnx": "d0c79912e8b7e54e0d5d817ca5d9c16ee6bfc762f49507b36acc7b7d727daea0", "shruti.json": "7451ca6be3732fc44eea85350378de06a1b1d56220695b2b4bac6a36947b701d" }, "onnx-int8": { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "a66f8d55fef125dedf8accb9bb3c7221eaa1af22063bbcbf411ef8cd5ad6f22f", "decoder.onnx": "5edf2c9e893afeaaa9de360f6d7a29bf30a9c8b27b8a2b6a9e0aa54897032836", "shruti.json": "cfdc6dafb393fda00c3ddac844cf18e9ff1362de99b1ea88e77c2ede044271c2" } },
  mni: { onnx: { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "300d19c3ff370dd25acecab98302cfa4b448f4b18cf1cc958243daded1d801f1", "decoder.onnx": "8450b44bb381a6606db2d9676a2570fe275ee975d53bdddeafe1f43c5356cf78", "shruti.json": "6e71e6dc4ae277a71a9bb428cca93499df00db2f357cdd8e972c6ce48a3361b8" }, "onnx-int8": { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "08dd3f6ca37b8b33141760b1878a634ed418a149ac7185a38a9e7567a9ca3bc5", "decoder.onnx": "bde6f2aa874b87b93a42fa4020df1022ca810dd37f87b134b4c22014f65ffb5e", "shruti.json": "96b9baedab86320d8c6806f2876795bee70fd6f44015b1df1656f1842e9df41b" } },
  mr: { onnx: { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "439babb5fa470f073d54d91c31275e0135106b3738b6fa00bf573534d4ebb57d", "decoder.onnx": "af648dc2b15d17afc2f4ea64ac1e3e27c4e14b3dc349b837e1b2525a1a935755", "shruti.json": "3f6da51f6a607b604a3209340088a2d5acb687c6ca0eefa7d38e0f99db131a08" }, "onnx-int8": { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "3092d2c5d28d650cfe3abb6bd713a3640ce169fda0eccb652a41f8236aa1e9df", "decoder.onnx": "5e03ba8f8b9d9f097f87466376610099345e10038204ffab003c397ca135b194", "shruti.json": "7ad66656fce7f71df16c0bbd6866a7a624c6599b5d9188675042da7c72645965" } },
  ne: { onnx: { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "e031998965228cf5e7fc2ee4079466ce7786b898e8313a38eb7d02029945be38", "decoder.onnx": "4a9c66dea702726dff9a4d1142029e2a51b00b5a491a6862b89d0114acd9ef7d", "shruti.json": "d61080fa58c775fb852b24fdc5f8362135fb04e7e74c8a691b8782b26f324ada" }, "onnx-int8": { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "4d82d5743b0c88d2ecd3564cbfdd59c5fa50731d25f7fbbd5dd2b5e51316449e", "decoder.onnx": "3b7a877b72ba951b1cad76ad0b18af67facb8ed7f09c00c5451ef39bea8eb2c5", "shruti.json": "911451514780c3904218ff9e89df88b521b4e63f02db1b6fdd8ae392a598e988" } },
  or: { onnx: { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "e38bfa59e399fc12acd063de8d9c4c15ed15c9dc130ec4e17cff73022333f2fa", "decoder.onnx": "20bf25c6b821001febf9f5329f19249a03e7ddbb4d3fabdd616234d05133b9f1", "shruti.json": "5b99be8a80d43c6178dd5550786bb406da81ba85b75da3eafc5c7db028ac2b7b" }, "onnx-int8": { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "072fe8837b08d0011fad5a4fa3d1c0ae3406aea1bb0628cc281ef35c00d5a12f", "decoder.onnx": "541b25f29ecc957333f7de9eddcba7273846291f09443e4dda9f6d292aa05406", "shruti.json": "2acd6aeb7e6754ae40a9221df4343a5956f12b9c92d4714321ad79caa588cd80" } },
  pa: { onnx: { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "7ae2896b77b1ebe4e9f34b6f42040a821afd49ab636ebfba51c107348ae59c21", "decoder.onnx": "5164a7eb2ae525e57e3619eecdb1f153ed64b22701acdea4a93b1e3e6313defc", "shruti.json": "8bf5a06e59a082bd4a4e004fac4d6534b2905e65be6accfacbb378df1bd2a21b" }, "onnx-int8": { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "6e1a4a5410631a3ef59ce55ce64a7387257f590dc5b4610e9a4ad25fe49f4d58", "decoder.onnx": "f4985b213fd3d77bbaa01b59282f8eb32e46a19be635271e06cd7a1338130453", "shruti.json": "1722f25e7d88cfa234538bea58e4e267f61c6f75c107966ad36c69d98e864204" } },
  sa: { onnx: { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "13c06fd1571be85d96220c6cf480ef0353b2de6dd42766417b4f1844a96950e9", "decoder.onnx": "a8e30bd50c498fe3b1908dc7582897eb0f8dd2b11cae7341d89b955acc378fa4", "shruti.json": "6fb59e5b6918172ca80fc0b1d5e1977e72cb64970ed72403c701a18d19c7ed30" }, "onnx-int8": { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "83e2a3277f25defaeb3459efe82d865d182ecd26f043d001dcd24a11df188417", "decoder.onnx": "81c0772cdb94304e27b3b80bb5e7dfcefda998e67b3b070b20312bf6d2e4cfe0", "shruti.json": "3330f0672a7a71f18cb8883bed80bda6cda8452a0e04bad928fd3bbbd32ed9aa" } },
  sat: { onnx: { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "9b1a5715f2d2a47d7595b1a6bc7f34d85fe637f3af6263b4c44434b554df6120", "decoder.onnx": "3961eacb54a19e1145568bdc7ebdea7ce865e19d74d09b5d91c1f5b93c14a667", "shruti.json": "bf8d7d698c1893dfe98f2d8c9e16a036ebe8bf0b6022b0effbb571ebcd9734be" }, "onnx-int8": { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "e48ea064ea707abc140d24a5e39ce40b4a8eda3f23e5ab0251b526d9f4773007", "decoder.onnx": "cf94a2e646177ca32d3d98c25408229a944510532510f00c2fc4fd9f19ebaf54", "shruti.json": "afc68fdb9a3b55a2c2949ae6e05a9495f80ffa03856e7d31c76df2ea4b1ccc4b" } },
  sd: { onnx: { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "e41d9133340c9835a68cf132fdc40a93dde8ca7f8f3e10a63aaba6285436e075", "decoder.onnx": "1d5f84ecf31de48ebe1ff67cf1a96705c67ea8f14a41dfafacb4a4301b169612", "shruti.json": "367c04f2684df92dfbd904747ac71d9ad91585a1450b34b2c491e2d62e248b75" }, "onnx-int8": { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "04d9db880b60a337b723ba28d2477cd39ff494bb98e690835e70659e444f3ec0", "decoder.onnx": "1f31c61b091fac9341311d55b8055546dcf7b578bd0f2aef97210d5145493240", "shruti.json": "ede61c64fe84c49f8665e4ee00a347289336e70fe7e76899472e82d6e8912fd0" } },
  ta: { onnx: { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "71b016a554116e6166fe9874530e4b486fb36cb695e386d8fa4c5665f8524530", "decoder.onnx": "abab0a54cc8d970a579121af2110a4cf18d38d7870ea2ead6fe6436dbf05f2d2", "shruti.json": "32c9339d032361a44c21d329e328d04b6a9008f3122d195911fc8ab8fa59e3c7" }, "onnx-int8": { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "4bb092c908b5b924d0c709616c77c3961d0a50baa19993b8380b182193618e16", "decoder.onnx": "89b9ee9bdaa449d97584741ef151f18ceedc958b2e18bcc79496097234b02647", "shruti.json": "82fca1b7e5bf9e27f9fae1395b723af7997f3615f6c70b2dc46339fdde37ed70" } },
  te: { onnx: { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "96aee89d5ad40d1e6e5f6338b4541f4c57d6c64bc3153320988b5ee89e3cefc4", "decoder.onnx": "48772a3ce46d69bba5f6cb6ddd2dae2b6939bbeeac2d7e54f733e7c3f1a6274a", "shruti.json": "6d6b1a6d0521858312b35e2a0a826ee745445c71c5c91860a60eeba29811627c" }, "onnx-int8": { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "8ce30588d47a14eb46a465bc5d40958c3a3050de55327c9ff82d5560cc8a58a5", "decoder.onnx": "d3f8cb9507bba6ecbfcee2fc3094631c7b3f38f28cdfb198f9d666807477591c", "shruti.json": "8a9c950c358229bad1b2653b27fbabefbe07bc5c2750a031c5a055d0438ca2f3" } },
  ur: { onnx: { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "daf48144654f8a63c588034e5ce5650ef9ae346100fba9e6b81722f9d65ff5d0", "decoder.onnx": "09fa2bdd19a7f6a80dc9a68842e9e300e62687ef900a504845b34f35f31b9548", "shruti.json": "7cb22c076685ab65c53d4e3b07f2ae3adc42a3678cc336cbe19861f98a2db81c" }, "onnx-int8": { "frontend.onnx": "0457d948c74b62e12b5d652d275269331092e13b22a2cdf64d2773c639a2b276", "encoder.onnx": "84c55771ee2d2fba71ae7938d1e743ec269a94bc6a09df18be3860b9a97ff9ed", "decoder.onnx": "a3203a1e7fb022829925a3c4c3f5e57b09537fbf40883a1c56e6f80ca5cbb25d", "shruti.json": "30d3bc1cb15dcae732cc0bcefcc6bcda3c040889b1c48748fc8f857ea37c5a8d" } },
};
const ALL_LANGUAGES = {
  as: "Assamese", bn: "Bengali", brx: "Bodo", doi: "Dogri", gu: "Gujarati", hi: "Hindi", kn: "Kannada", ks: "Kashmiri", kok: "Konkani", mai: "Maithili",
  ml: "Malayalam", mni: "Manipuri", mr: "Marathi", ne: "Nepali", or: "Odia", pa: "Punjabi", sa: "Sanskrit", sat: "Santali", sd: "Sindhi", ta: "Tamil", te: "Telugu", ur: "Urdu",
};
const RELEASED = ["as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"]; // set by set_release.py
export const LANGUAGES = Object.fromEntries(RELEASED.map((c) => [c, ALL_LANGUAGES[c]]));
const ALIASES = { bangla: "bn", oriya: "or", meitei: "mni", meetei: "mni", panjabi: "pa" };

const released = () => Object.entries(LANGUAGES).sort((a, b) => a[1].localeCompare(b[1])).map(([c, w]) => `${w} (${c})`).join(", ");

// "tamil", "Tamil" or "ta" -> "ta". Throws, listing the languages.
export function languageCode(name) {
  const key = String(name).trim().toLowerCase();
  if (Object.hasOwn(ALL_LANGUAGES, key)) return key;
  for (const [code, word] of Object.entries(ALL_LANGUAGES)) if (word.toLowerCase() === key) return code;
  if (Object.hasOwn(ALIASES, key)) return ALIASES[key];
  throw new Error(`unknown language ${JSON.stringify(name)}; choose one of: ${released()}`);
}

// A URL through fetch; in Node, a plain path through the file system.
async function defaultFetch(path) {
  const remote = typeof process === "undefined" || !process.versions?.node || /^[a-z]+:\/\//i.test(String(path));
  if (remote) return fetch(path);
  const fs = await import("node:fs/promises");
  const b = await fs.readFile(path);
  return { ok: true, status: 200, arrayBuffer: async () => b.buffer.slice(b.byteOffset, b.byteOffset + b.length), json: async () => JSON.parse(b.toString("utf8")) };
}

// SHA-256 of bytes, as lowercase hex; null if this environment has no Web Crypto (e.g. a page served over plain HTTP from
// somewhere other than localhost) and verification has to be skipped.
async function sha256Hex(bytes) {
  let subtle = globalThis.crypto?.subtle;
  if (!subtle) { try { subtle = (await import("node:crypto")).webcrypto.subtle; } catch { return null; } }
  const d = new Uint8Array(await subtle.digest("SHA-256", bytes));
  return Array.from(d, (b) => b.toString(16).padStart(2, "0")).join("");
}

export async function createShruti({ ort, language = null, variant = "4bit", baseUrl = null, fetchFn = defaultFetch, sessionOptions = {} }) {
  if (!ort) throw new Error('pass the runtime: import * as ort from "onnxruntime-web"; createShruti({ ort, language: "urdu" })');
  const code = language === null ? null : languageCode(language);
  if (!(variant in VARIANTS)) throw new Error(`variant must be one of ${Object.keys(VARIANTS).join(", ")}`);
  const releaseHub = baseUrl === null;
  if (baseUrl === null) {
    if (code === null) throw new Error('name a language: createShruti({ ort, language: "urdu" }), or pass baseUrl for your own copy');
    if (!Object.hasOwn(LANGUAGES, code)) throw new Error(`${ALL_LANGUAGES[code]} is not in this release; choose one of: ${released()}`);
    baseUrl = `${REPO}${code}/${VARIANTS[variant]}/`;
  } else if (!baseUrl.endsWith("/")) baseUrl += "/";
  // the published hashes for this exact language+variant, or undefined for a custom baseUrl (your own copy, SHRUTI_MODEL_DIR,
  // or the samples' ?local flag) — those have no release hash to check against, so they load unverified (see the warning below)
  const expected = releaseHub ? HASHES[code]?.[VARIANTS[variant]] : undefined;
  let warnedUnverified = false;
  const warnOnce = (msg) => { if (!warnedUnverified) { warnedUnverified = true; console.warn(`shruti-onnx: ${msg}`); } };
  const get = async (f) => {
    const r = await fetchFn(baseUrl + f);
    if (r.ok === false) throw new Error(`could not fetch ${baseUrl + f}: HTTP ${r.status}` + (r.status === 404 ? " (this language may not be in this release; see its table of languages)" : ""));
    const bytes = new Uint8Array(await r.arrayBuffer());
    const hash = expected?.[f];
    if (!hash) { warnOnce(`no published hash for ${baseUrl} — loading it unverified`); return bytes; }
    const got = await sha256Hex(bytes);
    if (got === null) { warnOnce("this environment has no Web Crypto to verify model hashes — loading unverified"); return bytes; }
    if (got !== hash) throw new Error(`${baseUrl + f} has sha256 ${got}, not the published ${hash} — refusing to load it (possible tampering, a stale mirror, or a corrupted download)`);
    return bytes;
  };
  const opt = { executionProviders: ["wasm"], graphOptimizationLevel: "all", ...sessionOptions };
  const load = async (f) => ort.InferenceSession.create(await get(f), opt);
  const [frontend, encoder, decoder, meta] = await Promise.all([
    load("frontend.onnx"), load("encoder.onnx"), load("decoder.onnx"), get("shruti.json").then((b) => JSON.parse(new TextDecoder().decode(b))),
  ]);
  const asr = new Shruti(ort, frontend, encoder, decoder, meta);
  if (code !== null && asr.languages[0] !== code) throw new Error(`${baseUrl} holds a ${asr.languages[0]} model, not ${code}`);
  return asr;
}

class Shruti {
  constructor(ort, frontend, encoder, decoder, meta) {
    Object.assign(this, { ort, frontend, encoder, decoder });
    this.cfg = meta.config;
    this.vocab = meta.vocab;
    this.text = this.cfg.text;
    this.bos = this.cfg.bos;
    this.eos = this.cfg.eos;
    this.tags = Object.fromEntries(this.cfg.languages.map((l, i) => [l, this.text + i]));
    // the model's own language and English: what it was trained on (the other tags are Cactus Whistle's)
    this.languages = [this.cfg.languages.find((l) => !["en", "de", "fr", "es", "it", "nl", "pl"].includes(l)), "en"];
    this.stateNames = decoder.inputNames.slice(2);
  }

  detok(ids) {
    let out = "", bytes = [];
    const flush = () => { if (bytes.length) { out += new TextDecoder().decode(new Uint8Array(bytes)); bytes = []; } };
    for (const i of ids) {
      const p = this.vocab[i];
      if (p.length === 6 && p.startsWith("<0x") && p.endsWith(">")) { bytes.push(parseInt(p.slice(3, 5), 16)); continue; }
      flush();
      if (!(p.startsWith("<") && p.endsWith(">"))) out += p;
    }
    flush();
    return out.replaceAll("▁", " ").trim();
  }

  initState(B) {
    const T = this.ort.Tensor, st = {};
    for (const n of this.stateNames) {
      if (n === "pos") st[n] = new T("int64", new BigInt64Array(1), [1]);
      else if (n === "hist") st[n] = new T("int64", new BigInt64Array(2 * B), [B, 2]);
      else if (/^tap\d/.test(n)) st[n] = new T("float32", new Float32Array(B * 2 * 608), [B, 2, 608]);
      else if (/^k\d/.test(n)) st[n] = new T("float32", new Float32Array(0), [B, 2, 0, this.cfg.qk_dim]);
      else if (/^v\d/.test(n)) st[n] = new T("float32", new Float32Array(0), [B, 2, 0, this.cfg.v_dim]);
      else st[n] = new T("float32", new Float32Array(B * 9 * 512), [B, 9, 512]);
    }
    return st;
  }

  // keep rows `src` of every state tensor (beam reordering); pos is shared
  pick(st, src) {
    const T = this.ort.Tensor, out = {};
    for (const [n, t] of Object.entries(st)) {
      if (n === "pos") { out[n] = t; continue; }
      const row = t.dims.slice(1).reduce((a, b) => a * b, 1);
      const data = new t.data.constructor(src.length * row);
      src.forEach((b, i) => data.set(t.data.subarray(b * row, (b + 1) * row), i * row));
      out[n] = new T(t.type, data, [src.length, ...t.dims.slice(1)]);
    }
    return out;
  }

  async step(tokens, cross, st) {
    const T = this.ort.Tensor;
    const r = await this.decoder.run({ tokens: new T("int64", BigInt64Array.from(tokens.flat(), BigInt), [tokens.length, tokens[0].length]), cross, ...st });
    const V = r.logits.dims[2], B = r.logits.dims[0], lg = [];
    for (let b = 0; b < B; b++) lg.push(Float32Array.from(r.logits.data.subarray(b * V, (b + 1) * V)));
    const next = {};
    for (const n of this.stateNames) next[n] = r["new_" + n];
    return [lg, next];
  }

  async encode(wav) {
    const T = this.ort.Tensor;
    const feats = (await this.frontend.run({ audio: new T("float32", wav, [1, wav.length]) })).features;
    const frames = new T("int64", BigInt64Array.from([BigInt(Math.floor(wav.length / this.cfg.hop))]), [1]);
    return (await this.encoder.run({ features: feats, frames })).cross;
  }

  static logSoftmax(x) {
    let m = -Infinity;
    for (const v of x) if (v > m) m = v;
    let s = 0;
    for (const v of x) s += Math.exp(v - m);
    const l = m + Math.log(s), out = new Float64Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = x[i] - l;
    return out;
  }

  static cutLoops(ids, maxPeriod = 8, repeats = 3) {
    const n = ids.length;
    for (let s = 0; s < n; s++) {
      for (let k = 1; k <= maxPeriod; k++) {
        if (s + k * repeats > n) break;
        let ok = true;
        for (let r = 1; r < repeats && ok; r++) for (let j = 0; j < k; j++) if (ids[s + r * k + j] !== ids[s + j]) { ok = false; break; }
        if (ok) return ids.slice(0, s + k);
      }
    }
    return ids;
  }

  // samples: Float32Array, 16 kHz mono. Options: language: null (detect between the model's own language and English), or either one by
  // word or code ("tamil", "ta", "english", "en"), beams 5 | 1,
  // numbers "words" | "digits" (Hindi model only), loops "off" (the model's output as it is) | "cut" (keep the first copy of a
  // repeated pattern and drop everything after it).
  async transcribe(samples, { language = null, beams = 5, numbers = "words", windowS = 30, lengthNorm = true, tokPerS = 12, loops = LOOPS_DEFAULT } = {}) {
    if (language !== null) {
      const key = String(language).trim().toLowerCase();
      language = key === "en" || key === "english" ? "en" : languageCode(language);
      if (!this.languages.includes(language)) throw new Error(`this is the ${LANGUAGES[this.languages[0]]} model: language must be "${this.languages[0]}", "en" or null to detect, not "${language}"`);
    }
    if (numbers === "digits" && this.languages[0] !== "hi") throw new Error('numbers: "digits" is tested for the Hindi model only');
    if (loops !== "off" && loops !== "cut") throw new Error('loops must be "off" or "cut"');
    let peak = 0;
    for (const v of samples) peak = Math.max(peak, Math.abs(v));
    if (samples.length === 0 || peak < 1e-4) return { text: "", language, tokens: [] };  // digital silence
    const n = Math.max(1, Math.ceil(samples.length / (windowS * RATE)));
    const parts = [], size = Math.floor(samples.length / n), extra = samples.length % n;
    for (let i = 0, start = 0; i < n; i++) {  // equal windows, like numpy.array_split
      const len = size + (i < extra ? 1 : 0);
      parts.push(await this._transcribe(samples.subarray(start, start + len), language, beams, tokPerS, lengthNorm, loops));
      start += len;
    }
    let text = parts.map((p) => p.text).filter(Boolean).join(" ");
    const lang = parts[0].language;
    if (numbers === "digits") {
      let norm;
      try { norm = await import("udgam-textnorm"); } catch (e) { throw new Error('numbers: "digits" needs the udgam-textnorm package, which is not on npm yet'); }
      text = norm.denormalize(text, lang, "digits");
    } else if (numbers !== "words") throw new Error('numbers must be "words" or "digits"');
    return { text, language: lang, tokens: parts.flatMap((p) => p.tokens) };
  }

  async _transcribe(wav, language, beams, tokPerS, lengthNorm, loops) {
    const tidy = loops === "cut" ? (ids) => Shruti.cutLoops(ids) : (ids) => ids;
    const cross = await this.encode(wav);
    const cap = Math.floor(Math.min(this.cfg.max_tokens - 2, (wav.length / RATE) * tokPerS + 8));
    let st = this.initState(1), lg;
    if (language === null) {
      [lg, st] = await this.step([[this.bos]], cross, st);
      language = this.languages.reduce((a, b) => (lg[0][this.tags[b]] > lg[0][this.tags[a]] ? b : a));
      [lg, st] = await this.step([[this.tags[language]]], cross, st);
    } else {
      [lg, st] = await this.step([[this.bos, this.tags[language]]], cross, st);
    }
    const mask = (rows) => rows.forEach((r) => r.fill(-Infinity, this.text));
    mask(lg);
    if (beams <= 1) {
      let ids = [];
      for (let i = 0; i < cap; i++) {
        let k = 0;
        for (let j = 1; j < this.text; j++) if (lg[0][j] > lg[0][k]) k = j;
        if (k === this.eos) break;
        ids.push(k);
        [lg, st] = await this.step([[k]], cross, st);
        mask(lg);
      }
      ids = tidy(ids);
      return { text: this.detok(ids), language, tokens: ids };
    }
    const norm = lengthNorm ? (sc, s) => sc / (s.length + 1) : (sc) => sc;
    const lp = Shruti.logSoftmax(lg[0]);
    const first = [...lp.keys()].sort((a, b) => lp[b] - lp[a] || a - b).slice(0, beams);
    let seqs = first.map((k) => [k]), scores = first.map((k) => lp[k]);
    const finished = [];
    seqs.forEach((s, i) => { if (s[s.length - 1] === this.eos) finished.push([scores[i], s.slice(0, -1)]); });
    const live = seqs.map((s, i) => i).filter((i) => seqs[i][seqs[i].length - 1] !== this.eos);
    seqs = live.map((i) => seqs[i]); scores = live.map((i) => scores[i]);
    st = this.pick(st, seqs.map(() => 0));
    for (let it = 0; it < cap - 1; it++) {
      if (!seqs.length) break;
      [lg, st] = await this.step(seqs.map((s) => [s[s.length - 1]]), cross, st);
      mask(lg);
      // the 2K best (beam, token) candidates, best first
      const top = [];
      lg.forEach((row, b) => {
        const l = Shruti.logSoftmax(row);
        for (let k = 0; k < l.length; k++) {
          const v = scores[b] + l[k];
          if (top.length < 2 * beams || v > top[top.length - 1][0]) {
            top.push([v, b, k]);
            top.sort((x, y) => y[0] - x[0] || x[1] * l.length + x[2] - (y[1] * l.length + y[2]));
            if (top.length > 2 * beams) top.pop();
          }
        }
      });
      const newSeqs = [], newScores = [], src = [];
      for (const [v, b, k] of top) {
        if (k === this.eos) finished.push([v, seqs[b]]);
        else { newSeqs.push([...seqs[b], k]); newScores.push(v); src.push(b); }
        if (newSeqs.length === beams) break;
      }
      const bestDone = finished.length ? Math.max(...finished.map(([sc, s]) => norm(sc, s))) : -Infinity;
      if (finished.length >= beams && (!newScores.length || Math.max(...newScores.map((sc, i) => norm(sc, newSeqs[i]))) < bestDone)) break;
      seqs = newSeqs; scores = newScores;
      st = this.pick(st, src);
    }
    const pool = finished.length ? finished : scores.map((sc, i) => [sc, seqs[i]]);
    let best = pool[0];
    for (const f of pool) if (norm(...f) > norm(...best)) best = f;
    const ids = tidy(best[1]);
    return { text: this.detok(ids), language, tokens: ids };
  }
}

// WAV (PCM 8/16/24/32-bit or float32) -> Float32Array at 16 kHz mono (linear resampling)
export function decodeWav(buf) {
  const dv = new DataView(buf);
  let p = 12, fmt = null, data = null;
  while (p + 8 <= dv.byteLength) {
    const id = String.fromCharCode(...new Uint8Array(buf, p, 4)), len = dv.getUint32(p + 4, true);
    if (id === "fmt ") fmt = { tag: dv.getUint16(p + 8, true), ch: dv.getUint16(p + 10, true), rate: dv.getUint32(p + 12, true), bits: dv.getUint16(p + 22, true) };
    if (id === "data") data = [p + 8, Math.min(len, dv.byteLength - p - 8)];
    p += 8 + len + (len % 2);
  }
  const bps = fmt.bits / 8, n = Math.floor(data[1] / bps / fmt.ch), x = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let c = 0; c < fmt.ch; c++) {
      const o = data[0] + (i * fmt.ch + c) * bps;
      if (fmt.tag === 3) s += dv.getFloat32(o, true);
      else if (bps === 1) s += (dv.getUint8(o) - 128) / 128;
      else if (bps === 2) s += dv.getInt16(o, true) / 32768;
      else if (bps === 3) s += ((dv.getUint8(o) | (dv.getUint8(o + 1) << 8) | (dv.getInt8(o + 2) << 16))) / 8388608;
      else s += dv.getInt32(o, true) / 2147483648;
    }
    x[i] = s / fmt.ch;
  }
  return resample(x, fmt.rate);
}

export function resample(x, rate) {
  if (rate === RATE) return x;
  const n = Math.round((x.length * RATE) / rate), y = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = (i * rate) / RATE, j = Math.floor(t), f = t - j;
    y[i] = j + 1 < x.length ? x[j] * (1 - f) + x[j + 1] * f : x[x.length - 1];
  }
  return y;
}
