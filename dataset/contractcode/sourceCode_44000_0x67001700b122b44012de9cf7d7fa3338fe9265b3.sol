/**
 *Submitted for verification at Etherscan.io on 2017-07-18
*/

pragma solidity ^0.4.8;
contract Counter {
  uint i=1;
  function inc() {
    i=i+1;
  }
  function get() constant returns (uint) {
    return i;
  }
}