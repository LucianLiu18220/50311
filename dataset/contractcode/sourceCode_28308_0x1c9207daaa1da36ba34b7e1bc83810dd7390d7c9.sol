/**
 *Submitted for verification at Etherscan.io on 2017-03-30
*/

contract ProofOfExistence {
  mapping (string => uint) private proofs;

  function storeProof(string sha256) {
    proofs[sha256] = block.timestamp;
  }

  function notarize(string sha256) {
    storeProof(sha256);
  }
  

  function checkDocument(string sha256) constant returns (uint) {
    return proofs[sha256];
  }
  
}